
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


# kernel path: /tmp/torchinductor_youkaichao/ss/cssgqgb7zuddcbhhjwwxeqb2asdek5xmxkdmui4zqbbsk6lqwf3l.py
# Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
# x_4 => clone, var_mean
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/24/c242qpbtt2vhmaiwchzntjqc5ogrdvihxri2fkj7msg3jnenzgxg.py
# Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
# x_4 => add, add_1, clone, mul, mul_1, rsqrt, sub, var_mean
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x0 = xindex % 3136
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (3136*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 128.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbdggom7aqjxgf26fxef3v6bkrncqx2mzht436rfbyragoqf4ag.py
# Source Nodes: [contiguous, shifted_x], Original ATen: [aten.clone, aten.native_layer_norm]
# contiguous => clone_1
# shifted_x => var_mean_1
triton_red_fused_clone_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    x4 = xindex % 7
    x5 = (xindex // 7) % 8
    x6 = (xindex // 56) % 7
    x8 = (xindex // 392)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 128.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (128*x4) + (896*x6) + (6272*x5) + (50176*x8)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chi3b37ge6x3jsbj67wjtkj5n4rw7ydpntnxffpb6et6elraywdb.py
# Source Nodes: [attn, q_1], Original ATen: [aten.clone, aten.mul]
# attn => clone_2
# q_1 => mul_4
triton_poi_fused_clone_mul_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvetzqnycltqdu4kpsua22rfc66vvnf4dehs4g32lvoiheddzc5a.py
# Source Nodes: [attn], Original ATen: [aten.clone]
# attn => clone_3
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (384*x2) + (18816*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttifggdho6egifzismnuabp4xrkfza7spfqjxqs63rc5hatjvaj.py
# Source Nodes: [attn_1, attn_2], Original ATen: [aten._softmax, aten.add]
# attn_1 => add_4
# attn_2 => amax, div, exp, sub_2, sum_1
triton_per_fused__softmax_add_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~rmask, "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (4*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cuji7ve3ttoamcvwwrtzsdozktqghta7owshka7hlchcidroa7jm.py
# Source Nodes: [x_7], Original ATen: [aten.clone]
# x_7 => clone_6
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 4
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (32*x2) + (384*x1) + (18816*x3)), None)
    tmp1 = tl.load(in_ptr1 + (256 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr64fuzx5xmbguxbfdgv33kbfdoh4iwiupkypbv7x2klmbw5otmc.py
# Source Nodes: [x_8], Original ATen: [aten.clone]
# x_8 => clone_7
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 4
    x2 = (xindex // 128) % 49
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (6272*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswu3gxoyp6uyzurw4ybcvgk4xhdou7nckps2c2sb3dgc4ut3qjc.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___0___blocks___0___norm2 => add_6, add_7, mul_5, mul_6, rsqrt_2, sub_3, var_mean_2
triton_red_fused_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5v36t4arlnggt37rtifkkgwrclqbmmm3pjybknzumesyglgddb.py
# Source Nodes: [x_17], Original ATen: [aten.gelu]
# x_17 => add_8, erf, mul_7, mul_8, mul_9
triton_poi_fused_gelu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_9', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czozjf3u2o3sf7wstyb3e6uyb4rbjhqkif4n6boo2cmvoqrc7vff.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm1, x_22], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_l__mod___layers___0___blocks___1___norm1 => var_mean_3
# x_22 => add_9
triton_per_fused_add_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (401408*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*((x0 % 56) % 7)) + (896*((x0 // 56) % 7)) + (6272*((x0 % 56) // 7)) + (50176*(x0 // 392)) + (401408*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/cod42gvqophjdgd5xtb73gks5o7iv357olprnmz7vuqyvcfvymyb.py
# Source Nodes: [contiguous_4], Original ATen: [aten.clone]
# contiguous_4 => clone_12
triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x5 = (xindex // 401408)
    x6 = (xindex // 128) % 56
    x7 = (xindex // 7168) % 56
    x2 = (xindex // 896) % 8
    x3 = (xindex // 7168) % 7
    x8 = xindex % 896
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (128*((3 + x6) % 56)) + (7168*((3 + x7) % 56)) + (401408*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((56*((3 + x7) % 56)) + (3136*x5) + ((3 + x6) % 56)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (896*x3) + (6272*x2) + (50176*x9)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7n7i6ajtrcjhoghqha72ahoawct6t2a2brw7j7ire3tp5zoqaad.py
# Source Nodes: [attn_8], Original ATen: [aten._softmax]
# attn_8 => amax_1, div_1, exp_1, sub_5, sum_2
triton_per_fused__softmax_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 4
    x2 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 64))), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~rmask, "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (4*tmp4)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pybnoxvbrxkadk7frtuil3w3humtpwivekosidiuj5t3dhoys4.py
# Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___0___blocks___1___norm2 => add_15, add_16, mul_13, mul_14, rsqrt_4, sub_6, var_mean_4
triton_red_fused_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*(((53 + (x0 % 56)) % 56) % 7)) + (896*(((53 + (x0 // 56)) % 56) % 7)) + (6272*(((53 + (x0 % 56)) % 56) // 7)) + (50176*(((53 + (x0 // 56)) % 56) // 7)) + (401408*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + (128*(((53 + (x0 % 56)) % 56) % 7)) + (896*(((53 + (x0 // 56)) % 56) % 7)) + (6272*(((53 + (x0 % 56)) % 56) // 7)) + (50176*(((53 + (x0 // 56)) % 56) // 7)) + (401408*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 128.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = tl.math.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7o/c7optdjgxswjb43u4d5mvkeb3r2u7sigmnpvp5nyyzhbpidyuegr.py
# Source Nodes: [x_43], Original ATen: [aten.clone]
# x_43 => clone_23
triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(((53 + x1) % 56) % 7)) + (896*(((53 + x2) % 56) % 7)) + (6272*(((53 + x1) % 56) // 7)) + (50176*(((53 + x2) % 56) // 7)) + (401408*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2iokhhxush3kfukqkucklkr2lftaton4xyz5gtxcylzywf3u6x.py
# Source Nodes: [x_44], Original ATen: [aten.native_layer_norm]
# x_44 => add_19, add_20, mul_18, mul_19, rsqrt_5, sub_7, var_mean_5
triton_per_fused_native_layer_norm_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(r2 // 256)) + (256*x0) + (7168*((r2 // 128) % 2)) + (14336*x1) + (r2 % 128)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv52lyvzbeicp25uo5lzuy37x3ki4ul4vl2db4zxz6ikhmc7m2je.py
# Source Nodes: [contiguous_8, shifted_x_8], Original ATen: [aten.clone, aten.native_layer_norm]
# contiguous_8 => clone_24
# shifted_x_8 => var_mean_6
triton_per_fused_clone_native_layer_norm_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    x2 = xindex % 7
    x3 = (xindex // 7) % 4
    x4 = (xindex // 28) % 7
    x5 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 256.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (256*x2) + (1792*x4) + (12544*x3) + (50176*x5)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23xnrtyt6slhqvk2x3w2dnkpnr6ula2xkpmp77joxqs5ea2d7ix.py
# Source Nodes: [attn_10, q_5], Original ATen: [aten.clone, aten.mul]
# attn_10 => clone_25
# q_5 => mul_22
triton_poi_fused_clone_mul_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/car6r4e3cl22yusr35wdvl2qudibfv2qi6a7yhogynjusufuyayt.py
# Source Nodes: [attn_10], Original ATen: [aten.clone]
# attn_10 => clone_26
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (256 + y0 + (768*x2) + (37632*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/cheavxkyd46n5jgq3xohqhkwu4x3ksxrhelb54g4sjy5hs6wh7ml.py
# Source Nodes: [attn_11, attn_12], Original ATen: [aten._softmax, aten.add]
# attn_11 => add_23
# attn_12 => amax_2, div_2, exp_2, sub_9, sum_3
triton_per_fused__softmax_add_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (8*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyy2ihc7aibetj6e26i2hfzojt755vo6vxqjhgv3dgdo2xo3hmhn.py
# Source Nodes: [x_48], Original ATen: [aten.clone]
# x_48 => clone_29
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (37632*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvye7pr5uqm3dempa46btroo5i7552wzhxbpf4cnof3gqqlha7h.py
# Source Nodes: [x_49], Original ATen: [aten.clone]
# x_49 => clone_30
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 8
    x2 = (xindex // 256) % 49
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (12544*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqrbcnbrykxt4qxldihdbsfirlud3bqdyf7j2abosylc55hurjt.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___1___blocks___0___norm2 => add_25, add_26, mul_23, mul_24, rsqrt_7, sub_10, var_mean_7
triton_per_fused_native_layer_norm_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_22', 'mutated_arg_names': []}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7sfmxdzy2beozsmxr26qc4ewwzpgxxzfzjes65tgycppfw6dit.py
# Source Nodes: [x_58], Original ATen: [aten.gelu]
# x_58 => add_27, erf_2, mul_25, mul_26, mul_27
triton_poi_fused_gelu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_23', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkzcwd2phyuoaclugjzbhzy4mbovfxr7alwwamzwjra4s4mwaiv.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm1, x_63], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_l__mod___layers___1___blocks___1___norm1 => var_mean_8
# x_63 => add_28
triton_per_fused_add_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*((x0 % 28) % 7)) + (1792*((x0 // 28) % 7)) + (12544*((x0 % 28) // 7)) + (50176*(x0 // 196)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 256, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu22rfwaaxubr24vzrkzik2xotcijjvt3hxrh6v7kb56ua2ixjwq.py
# Source Nodes: [contiguous_12], Original ATen: [aten.clone]
# contiguous_12 => clone_35
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x5 = (xindex // 200704)
    x6 = (xindex // 256) % 28
    x7 = (xindex // 7168) % 28
    x2 = (xindex // 1792) % 4
    x3 = (xindex // 7168) % 7
    x8 = xindex % 1792
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((3 + x6) % 28)) + (7168*((3 + x7) % 28)) + (200704*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (1792*x3) + (12544*x2) + (50176*x9)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cylrtmdlpvl2tzs3aswm73ugwbhziycf2yuwscvnlhgeegkgjm5j.py
# Source Nodes: [attn_18], Original ATen: [aten._softmax]
# attn_18 => amax_3, div_3, exp_3, sub_12, sum_4
triton_per_fused__softmax_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    x2 = (xindex // 392)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 16))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (8*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7egampwcexyyl5yjrsvfilyd64mag3iobo6xhdgwzd672vhpqz.py
# Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___1___blocks___1___norm2 => add_34, add_35, mul_31, mul_32, rsqrt_9, sub_13, var_mean_9
triton_per_fused_native_layer_norm_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_27', 'mutated_arg_names': []}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*(((25 + (x0 % 28)) % 28) % 7)) + (1792*(((25 + (x0 // 28)) % 28) % 7)) + (12544*(((25 + (x0 % 28)) % 28) // 7)) + (50176*(((25 + (x0 // 28)) % 28) // 7)) + (200704*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l5wdn7rply5zmskqaegm6qvso43ovlyruh626yzkplwvp5xlv5.py
# Source Nodes: [x_84], Original ATen: [aten.clone]
# x_84 => clone_46
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(((25 + x1) % 28) % 7)) + (1792*(((25 + x2) % 28) % 7)) + (12544*(((25 + x1) % 28) // 7)) + (50176*(((25 + x2) % 28) // 7)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2yw5bg7mnrogrjumfbhazqppys6xobaht55e7pky2igvvdlmtf.py
# Source Nodes: [x_85], Original ATen: [aten.native_layer_norm]
# x_85 => add_38, add_39, mul_36, mul_37, rsqrt_10, sub_14, var_mean_10
triton_per_fused_native_layer_norm_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 14
    x1 = (xindex // 14)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(r2 // 512)) + (512*x0) + (7168*((r2 // 256) % 2)) + (14336*x1) + (r2 % 256)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgrvvnmclb67dxjtzj7uvbcuabrqlssp5j2u376w7gau7crncld.py
# Source Nodes: [contiguous_16, shifted_x_16], Original ATen: [aten.clone, aten.native_layer_norm]
# contiguous_16 => clone_47
# shifted_x_16 => var_mean_11
triton_per_fused_clone_native_layer_norm_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
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
    x2 = xindex % 7
    x3 = (xindex // 7) % 2
    x4 = (xindex // 14) % 7
    x5 = (xindex // 98)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (512*x2) + (3584*x4) + (25088*x3) + (50176*x5)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bb5gr5p5vejj6n5qqyw2snuuzcv4frke6hroxc3l4plmmtok5s.py
# Source Nodes: [attn_20, q_9], Original ATen: [aten.clone, aten.mul]
# attn_20 => clone_48
# q_9 => mul_40
triton_poi_fused_clone_mul_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5g5jhizxguylfndqikpuymy5vcf5hkwsl5nqqcrx3izu3bz75z.py
# Source Nodes: [attn_20], Original ATen: [aten.clone]
# attn_20 => clone_49
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3la5l6mvljcksoqpqkhhor7en6rwwbkuktkjkgxapbib5xq42u.py
# Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.add]
# attn_21 => add_42
# attn_22 => amax_4, div_4, exp_4, sub_16, sum_5
triton_per_fused__softmax_add_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (16*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbvsek6dfoq4575baot5y2uuabj4bnaygiue2ecr5afb2jfr3lw.py
# Source Nodes: [x_89], Original ATen: [aten.clone]
# x_89 => clone_52
triton_poi_fused_clone_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 16
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (32*x2) + (1536*x1) + (75264*x3)), None)
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpsxp34fmoex5i27qksrxh5ss5twfjpiez4f5y4m6pazlrur3xq.py
# Source Nodes: [x_90], Original ATen: [aten.clone]
# x_90 => clone_53
triton_poi_fused_clone_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 49
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (25088*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4uea5tx3xakqkz22f4wc4v2vex56voab7lywplgo46anlgghgqi.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___2___blocks___0___norm2 => add_44, add_45, mul_41, mul_42, rsqrt_12, sub_17, var_mean_12
triton_per_fused_native_layer_norm_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_36', 'mutated_arg_names': []}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnmeiyeabdq4pql7etksixf4n3czmbwda3r5rlybhws4kqa32nh.py
# Source Nodes: [x_99], Original ATen: [aten.gelu]
# x_99 => add_46, erf_4, mul_43, mul_44, mul_45
triton_poi_fused_gelu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_37', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdr73w6xxdjthssurzg4dpqf3wuowlxfkcxti5sbahm5fxzveiy.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm1, x_104], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_l__mod___layers___2___blocks___1___norm1 => var_mean_13
# x_104 => add_47
triton_per_fused_add_native_layer_norm_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_38', 'mutated_arg_names': ['in_out_ptr0']}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*((x0 % 14) % 7)) + (3584*((x0 // 14) % 7)) + (25088*((x0 % 14) // 7)) + (50176*(x0 // 98)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcnzh3v2fq6kwo3lqyx5ki6zs4eurll6hiplb7o5qz3yusqoqt4.py
# Source Nodes: [contiguous_20], Original ATen: [aten.clone]
# contiguous_20 => clone_58
triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x5 = (xindex // 100352)
    x6 = (xindex // 512) % 14
    x7 = (xindex // 7168) % 14
    x2 = (xindex // 3584) % 2
    x3 = (xindex // 7168) % 7
    x8 = xindex % 3584
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((3 + x6) % 14)) + (7168*((3 + x7) % 14)) + (100352*x5)), None)
    tmp1 = tl.load(in_ptr1 + ((14*((3 + x7) % 14)) + (196*x5) + ((3 + x6) % 14)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((14*((3 + x7) % 14)) + (196*x5) + ((3 + x6) % 14)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x8 + (3584*x3) + (25088*x2) + (50176*x9)), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmyfoyakopg6siowlpziatjllnjwji6uoushy4lu446ak33av4w.py
# Source Nodes: [attn_28], Original ATen: [aten._softmax]
# attn_28 => amax_5, div_5, exp_5, sub_19, sum_6
triton_per_fused__softmax_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 16
    x2 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r3 + (49*x0) + (2401*(x2 % 4))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (16*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr3 + (r3 + (49*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c5537kxtni7lela72iwejdc34n2l26jm247awdlnvz2g4jp7kygw.py
# Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___2___blocks___1___norm2 => add_53, add_54, mul_49, mul_50, rsqrt_14, sub_20, var_mean_14
triton_per_fused_native_layer_norm_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_41', 'mutated_arg_names': []}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2ptdqjm5cosnog22ha5swmsdltwkxqpiwhye25luvwds252f2b.py
# Source Nodes: [contiguous_24, shifted_x_24, x_122], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# contiguous_24 => clone_69
# shifted_x_24 => var_mean_15
# x_122 => add_56
triton_per_fused_add_clone_native_layer_norm_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_42', 'mutated_arg_names': ['in_out_ptr0']}
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x4 = xindex % 7
    x5 = (xindex // 7) % 2
    x6 = (xindex // 14) % 7
    x7 = (xindex // 98)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*(((11 + (x0 % 14)) % 14) % 7)) + (3584*(((11 + (x0 // 14)) % 14) % 7)) + (25088*(((11 + (x0 % 14)) % 14) // 7)) + (50176*(((11 + (x0 // 14)) % 14) // 7)) + (100352*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x4) + (3584*x6) + (25088*x5) + (50176*x7)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fy/cfyb3zktnlgwogagyd7z5xknenakqxgtojjfgopuvpycko5qdxgz.py
# Source Nodes: [x_413], Original ATen: [aten.clone]
# x_413 => clone_245
triton_poi_fused_clone_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 14
    x2 = (xindex // 7168) % 14
    x3 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*(((11 + x1) % 14) % 7)) + (3584*(((11 + x2) % 14) % 7)) + (25088*(((11 + x1) % 14) // 7)) + (50176*(((11 + x2) % 14) // 7)) + (100352*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x4), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cgl2lgqpchtqfk55nhxvf6omrcdwfrxvarcv2svwn26oelbwlvqg.py
# Source Nodes: [x_414], Original ATen: [aten.native_layer_norm]
# x_414 => add_193, add_194, mul_182, mul_183, rsqrt_47, sub_69, var_mean_47
triton_red_fused_native_layer_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((512*(r2 // 1024)) + (1024*x0) + (7168*((r2 // 512) % 2)) + (14336*x1) + (r2 % 512)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 2048.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cilzffob3o6baa2fpqhemnlt3leqijvqexyhaomhzzogqhalwqsq.py
# Source Nodes: [shifted_x_88], Original ATen: [aten.native_layer_norm]
# shifted_x_88 => add_195, add_196, mul_184, mul_185, rsqrt_48, sub_70, var_mean_48
triton_per_fused_native_layer_norm_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cnt3aqdpnb2mu22fhw4bbhhfx24pn7ox46265wvir3e4fqc6h4pm.py
# Source Nodes: [attn_110, q_45], Original ATen: [aten.clone, aten.mul]
# attn_110 => clone_246
# q_45 => mul_186
triton_poi_fused_clone_mul_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2vnc7zhbkytqqo5j764xmlkmn2jasuvdqoqjxdrundura6pde7.py
# Source Nodes: [attn_110], Original ATen: [aten.clone]
# attn_110 => clone_247
triton_poi_fused_clone_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (1024 + y0 + (3072*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cikwbapmmct4vzlqogcwibbeusueezmxap7ctrxoc5q6ogozgqvk.py
# Source Nodes: [attn_111, attn_112], Original ATen: [aten._softmax, aten.add]
# attn_111 => add_197
# attn_112 => amax_22, div_22, exp_22, sub_71, sum_23
triton_per_fused__softmax_add_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 32
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 + 169
    tmp3 = tmp1 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp1)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 169)) | ~(xmask & rmask), "index out of bounds: 0 <= tmp4 < 169")
    tmp5 = tl.load(in_ptr2 + (x1 + (32*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp12 / tmp16
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp17, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjiynbvgf6ox2bqhgp3hmushksspvmcbtynp7mjd37voddpwskt.py
# Source Nodes: [x_418], Original ATen: [aten.clone]
# x_418 => clone_250
triton_poi_fused_clone_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 32
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (32*x2) + (3072*x1) + (150528*x3)), None)
    tmp1 = tl.load(in_ptr1 + (2048 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5r6oqmhgmn7b2ufe5kmfjuchlom2ihi63ihs2vcsyug4fjgbb5.py
# Source Nodes: [x_419], Original ATen: [aten.clone]
# x_419 => clone_251
triton_poi_fused_clone_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 49
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1568*x1) + (50176*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbvgdl4sdzpcdy737glxsqebqfh3ebpidshoykt752l7ogvb2x6.py
# Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
# getattr_getattr_l__mod___layers___3___blocks___0___norm2 => add_199, add_200, mul_187, mul_188, rsqrt_49, sub_72, var_mean_49
triton_per_fused_native_layer_norm_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 1024, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 1024.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cbl2jtbpzaovy6gokqsn3iwsfqwqiaaecryhpn7c4nckt7gi4y75.py
# Source Nodes: [x_428], Original ATen: [aten.gelu]
# x_428 => add_201, erf_22, mul_189, mul_190, mul_191
triton_poi_fused_gelu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxa3u6dl77ycbb7mabrfcqrj2zay2eebuqjaqsamysipuk6mx7t.py
# Source Nodes: [shifted_x_92, x_433], Original ATen: [aten.add, aten.native_layer_norm]
# shifted_x_92 => add_203, add_204, mul_192, mul_193, rsqrt_50, sub_73, var_mean_50
# x_433 => add_202
triton_per_fused_add_native_layer_norm_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 1024.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl43ppqogq3kjrbwkjeqzdvcqtdkwq4xnbg6gk5cnma4m2javoad.py
# Source Nodes: [x_451, x_456], Original ATen: [aten.add, aten.native_layer_norm]
# x_451 => add_210
# x_456 => var_mean_52
triton_per_fused_add_native_layer_norm_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 1024, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmxlf4v34goccl46wvd53ss5tbid4jxbbnwlq5rvwnc6gfk5wwh.py
# Source Nodes: [x_456, x_457], Original ATen: [aten.mean, aten.native_layer_norm]
# x_456 => add_211, add_212, mul_200, mul_201, rsqrt_52, sub_76, var_mean_52
# x_457 => mean
triton_per_fused_mean_native_layer_norm_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_native_layer_norm_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (50176*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 49.0
    tmp19 = tmp17 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
    args.clear()
    assert_size_stride(arg0_1, (169, 4), (4, 1))
    assert_size_stride(arg1_1, (169, 4), (4, 1))
    assert_size_stride(arg2_1, (169, 8), (8, 1))
    assert_size_stride(arg3_1, (169, 8), (8, 1))
    assert_size_stride(arg4_1, (169, 16), (16, 1))
    assert_size_stride(arg5_1, (169, 16), (16, 1))
    assert_size_stride(arg6_1, (169, 16), (16, 1))
    assert_size_stride(arg7_1, (169, 16), (16, 1))
    assert_size_stride(arg8_1, (169, 16), (16, 1))
    assert_size_stride(arg9_1, (169, 16), (16, 1))
    assert_size_stride(arg10_1, (169, 16), (16, 1))
    assert_size_stride(arg11_1, (169, 16), (16, 1))
    assert_size_stride(arg12_1, (169, 16), (16, 1))
    assert_size_stride(arg13_1, (169, 16), (16, 1))
    assert_size_stride(arg14_1, (169, 16), (16, 1))
    assert_size_stride(arg15_1, (169, 16), (16, 1))
    assert_size_stride(arg16_1, (169, 16), (16, 1))
    assert_size_stride(arg17_1, (169, 16), (16, 1))
    assert_size_stride(arg18_1, (169, 16), (16, 1))
    assert_size_stride(arg19_1, (169, 16), (16, 1))
    assert_size_stride(arg20_1, (169, 16), (16, 1))
    assert_size_stride(arg21_1, (169, 16), (16, 1))
    assert_size_stride(arg22_1, (169, 32), (32, 1))
    assert_size_stride(arg23_1, (169, 32), (32, 1))
    assert_size_stride(arg24_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (384, 128), (128, 1))
    assert_size_stride(arg31_1, (384, ), (1, ))
    assert_size_stride(arg32_1, (128, 128), (128, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (512, 128), (128, 1))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (128, 512), (512, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (384, 128), (128, 1))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (128, 128), (128, 1))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, 128), (128, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (128, 512), (512, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (256, 512), (512, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (768, 256), (256, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (256, 256), (256, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (1024, 256), (256, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (256, 1024), (1024, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (768, 256), (256, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (256, 256), (256, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (1024, 256), (256, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (256, 1024), (1024, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (512, 1024), (1024, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (1536, 512), (512, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (2048, 512), (512, 1))
    assert_size_stride(arg91_1, (2048, ), (1, ))
    assert_size_stride(arg92_1, (512, 2048), (2048, 1))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (1536, 512), (512, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (2048, 512), (512, 1))
    assert_size_stride(arg103_1, (2048, ), (1, ))
    assert_size_stride(arg104_1, (512, 2048), (2048, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (1536, 512), (512, 1))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (512, 512), (512, 1))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (2048, 512), (512, 1))
    assert_size_stride(arg115_1, (2048, ), (1, ))
    assert_size_stride(arg116_1, (512, 2048), (2048, 1))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (1536, 512), (512, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (2048, 512), (512, 1))
    assert_size_stride(arg127_1, (2048, ), (1, ))
    assert_size_stride(arg128_1, (512, 2048), (2048, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (1536, 512), (512, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (512, 512), (512, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (2048, 512), (512, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (512, 2048), (2048, 1))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (1536, 512), (512, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (512, 512), (512, 1))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (2048, 512), (512, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (512, 2048), (2048, 1))
    assert_size_stride(arg153_1, (512, ), (1, ))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (1536, 512), (512, 1))
    assert_size_stride(arg157_1, (1536, ), (1, ))
    assert_size_stride(arg158_1, (512, 512), (512, 1))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (2048, 512), (512, 1))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (512, 2048), (2048, 1))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (1536, 512), (512, 1))
    assert_size_stride(arg169_1, (1536, ), (1, ))
    assert_size_stride(arg170_1, (512, 512), (512, 1))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (2048, 512), (512, 1))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (512, 2048), (2048, 1))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (1536, 512), (512, 1))
    assert_size_stride(arg181_1, (1536, ), (1, ))
    assert_size_stride(arg182_1, (512, 512), (512, 1))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (2048, 512), (512, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (512, 2048), (2048, 1))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, ), (1, ))
    assert_size_stride(arg192_1, (1536, 512), (512, 1))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (512, 512), (512, 1))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, ), (1, ))
    assert_size_stride(arg198_1, (2048, 512), (512, 1))
    assert_size_stride(arg199_1, (2048, ), (1, ))
    assert_size_stride(arg200_1, (512, 2048), (2048, 1))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (1536, 512), (512, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (512, 512), (512, 1))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (2048, 512), (512, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (512, 2048), (2048, 1))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (1536, 512), (512, 1))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (512, 512), (512, 1))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (2048, 512), (512, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (512, 2048), (2048, 1))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (1536, 512), (512, 1))
    assert_size_stride(arg229_1, (1536, ), (1, ))
    assert_size_stride(arg230_1, (512, 512), (512, 1))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (2048, 512), (512, 1))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (512, 2048), (2048, 1))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (1536, 512), (512, 1))
    assert_size_stride(arg241_1, (1536, ), (1, ))
    assert_size_stride(arg242_1, (512, 512), (512, 1))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (2048, 512), (512, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (512, 2048), (2048, 1))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, ), (1, ))
    assert_size_stride(arg252_1, (1536, 512), (512, 1))
    assert_size_stride(arg253_1, (1536, ), (1, ))
    assert_size_stride(arg254_1, (512, 512), (512, 1))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (2048, 512), (512, 1))
    assert_size_stride(arg259_1, (2048, ), (1, ))
    assert_size_stride(arg260_1, (512, 2048), (2048, 1))
    assert_size_stride(arg261_1, (512, ), (1, ))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (1536, 512), (512, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (512, 512), (512, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (2048, 512), (512, 1))
    assert_size_stride(arg271_1, (2048, ), (1, ))
    assert_size_stride(arg272_1, (512, 2048), (2048, 1))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (1536, 512), (512, 1))
    assert_size_stride(arg277_1, (1536, ), (1, ))
    assert_size_stride(arg278_1, (512, 512), (512, 1))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, ), (1, ))
    assert_size_stride(arg282_1, (2048, 512), (512, 1))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (512, 2048), (2048, 1))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (1536, 512), (512, 1))
    assert_size_stride(arg289_1, (1536, ), (1, ))
    assert_size_stride(arg290_1, (512, 512), (512, 1))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (2048, 512), (512, 1))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (512, 2048), (2048, 1))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (2048, ), (1, ))
    assert_size_stride(arg299_1, (2048, ), (1, ))
    assert_size_stride(arg300_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg304_1, (3072, ), (1, ))
    assert_size_stride(arg305_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg310_1, (4096, ), (1, ))
    assert_size_stride(arg311_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg316_1, (3072, ), (1, ))
    assert_size_stride(arg317_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg322_1, (4096, ), (1, ))
    assert_size_stride(arg323_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1000, ), (1, ))
    assert_size_stride(arg329_1, (49, 49), (49, 1))
    assert_size_stride(arg330_1, (64, 49, 49), (2401, 49, 1))
    assert_size_stride(arg331_1, (49, 49), (49, 1))
    assert_size_stride(arg332_1, (49, 49), (49, 1))
    assert_size_stride(arg333_1, (16, 49, 49), (2401, 49, 1))
    assert_size_stride(arg334_1, (49, 49), (49, 1))
    assert_size_stride(arg335_1, (49, 49), (49, 1))
    assert_size_stride(arg336_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg337_1, (49, 49), (49, 1))
    assert_size_stride(arg338_1, (49, 49), (49, 1))
    assert_size_stride(arg339_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg340_1, (49, 49), (49, 1))
    assert_size_stride(arg341_1, (49, 49), (49, 1))
    assert_size_stride(arg342_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg343_1, (49, 49), (49, 1))
    assert_size_stride(arg344_1, (49, 49), (49, 1))
    assert_size_stride(arg345_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg346_1, (49, 49), (49, 1))
    assert_size_stride(arg347_1, (49, 49), (49, 1))
    assert_size_stride(arg348_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg349_1, (49, 49), (49, 1))
    assert_size_stride(arg350_1, (49, 49), (49, 1))
    assert_size_stride(arg351_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg352_1, (49, 49), (49, 1))
    assert_size_stride(arg353_1, (49, 49), (49, 1))
    assert_size_stride(arg354_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg355_1, (49, 49), (49, 1))
    assert_size_stride(arg356_1, (49, 49), (49, 1))
    assert_size_stride(arg357_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg358_1, (49, 49), (49, 1))
    assert_size_stride(arg359_1, (49, 49), (49, 1))
    assert_size_stride(arg360_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg361_1, (49, 49), (49, 1))
    assert_size_stride(arg362_1, (49, 49), (49, 1))
    assert_size_stride(arg363_1, (49, 49), (49, 1))
    assert_size_stride(arg364_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg364_1, arg24_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg24_1
        del arg364_1
        buf1 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_cuda_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf0, arg25_1, buf1, buf2, 25088, 128, grid=grid(25088), stream=stream0)
        buf4 = empty_strided((8, 56, 56, 128), (401408, 56, 1, 3136), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_1.run(buf0, arg25_1, buf1, buf2, arg26_1, arg27_1, buf4, 3211264, grid=grid(3211264), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        buf8 = reinterpret_tensor(buf0, (8, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), 0); del buf0  # reuse
        # Source Nodes: [contiguous, shifted_x], Original ATen: [aten.clone, aten.native_layer_norm]
        triton_red_fused_clone_native_layer_norm_2.run(buf4, arg28_1, arg29_1, buf8, 25088, 128, grid=grid(25088), stream=stream0)
        del arg28_1
        del arg29_1
        buf9 = empty((25088, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (25088, 128), (128, 1), 0), reinterpret_tensor(arg30_1, (128, 384), (1, 128), 0), out=buf9)
        del arg30_1
        buf10 = reinterpret_tensor(buf8, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf8  # reuse
        # Source Nodes: [attn, q_1], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_3.run(buf9, arg31_1, buf10, 3211264, grid=grid(3211264), stream=stream0)
        buf11 = empty((512, 4, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf9, arg31_1, buf11, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf12 = empty((2048, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf11, (2048, 32, 49), (1568, 49, 1), 0), out=buf12)
        buf15 = empty((512, 4, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_1, attn_2], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_5.run(buf12, arg329_1, arg0_1, buf15, 100352, 49, grid=grid(100352), stream=stream0)
        del arg0_1
        del arg329_1
        buf16 = reinterpret_tensor(buf11, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf11  # reuse
        # Source Nodes: [x_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf9, arg31_1, buf16, 3211264, grid=grid(3211264), stream=stream0)
        del arg31_1
        buf17 = reinterpret_tensor(buf10, (2048, 49, 32), (1568, 32, 1), 0); del buf10  # reuse
        # Source Nodes: [x_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf16, (2048, 49, 32), (1568, 32, 1), 0), out=buf17)
        buf18 = reinterpret_tensor(buf16, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf16  # reuse
        # Source Nodes: [x_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf17, buf18, 3211264, grid=grid(3211264), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (25088, 128), (128, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (25088, 128), (128, 1), 0), reinterpret_tensor(arg32_1, (128, 128), (1, 128), 0), out=buf19)
        del arg32_1
        buf23 = reinterpret_tensor(buf18, (8, 3136, 128), (401408, 128, 1), 0); del buf18  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_8.run(buf4, buf19, arg33_1, arg34_1, arg35_1, buf23, 25088, 128, grid=grid(25088), stream=stream0)
        del arg34_1
        del arg35_1
        buf24 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (25088, 128), (128, 1), 0), reinterpret_tensor(arg36_1, (128, 512), (1, 128), 0), out=buf24)
        del arg36_1
        buf25 = reinterpret_tensor(buf24, (8, 3136, 512), (1605632, 512, 1), 0); del buf24  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_9.run(buf25, arg37_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg37_1
        buf26 = reinterpret_tensor(buf23, (25088, 128), (128, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (25088, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 128), (1, 512), 0), out=buf26)
        del arg38_1
        buf27 = reinterpret_tensor(buf26, (8, 3136, 128), (401408, 128, 1), 0); del buf26  # reuse
        buf28 = buf2; del buf2  # reuse
        buf29 = buf1; del buf1  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm1, x_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_10.run(buf27, buf4, buf19, arg33_1, arg39_1, buf28, buf29, 25088, 128, grid=grid(25088), stream=stream0)
        del arg33_1
        del arg39_1
        buf31 = reinterpret_tensor(buf4, (8, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), 0); del buf4  # reuse
        # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf27, buf28, buf29, arg40_1, arg41_1, buf31, 3211264, grid=grid(3211264), stream=stream0)
        del arg40_1
        del arg41_1
        del buf28
        del buf29
        buf32 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (25088, 128), (128, 1), 0), reinterpret_tensor(arg42_1, (128, 384), (1, 128), 0), out=buf32)
        del arg42_1
        buf33 = reinterpret_tensor(buf31, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf31  # reuse
        # Source Nodes: [attn_4, q_3], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_3.run(buf32, arg43_1, buf33, 3211264, grid=grid(3211264), stream=stream0)
        buf34 = reinterpret_tensor(buf19, (512, 4, 32, 49), (6272, 1568, 49, 1), 0); del buf19  # reuse
        # Source Nodes: [attn_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf32, arg43_1, buf34, 65536, 49, grid=grid(65536, 49), stream=stream0)
        buf35 = reinterpret_tensor(buf15, (2048, 49, 49), (2401, 49, 1), 0); del buf15  # reuse
        # Source Nodes: [attn_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf34, (2048, 32, 49), (1568, 49, 1), 0), out=buf35)
        buf39 = reinterpret_tensor(buf12, (512, 4, 49, 49), (9604, 2401, 49, 1), 0); del buf12  # reuse
        # Source Nodes: [attn_8], Original ATen: [aten._softmax]
        triton_per_fused__softmax_12.run(buf35, arg331_1, arg1_1, arg330_1, buf39, 100352, 49, grid=grid(100352), stream=stream0)
        del arg1_1
        del arg330_1
        del arg331_1
        del buf35
        buf40 = reinterpret_tensor(buf34, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf34  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf32, arg43_1, buf40, 3211264, grid=grid(3211264), stream=stream0)
        del arg43_1
        del buf32
        buf41 = reinterpret_tensor(buf33, (2048, 49, 32), (1568, 32, 1), 0); del buf33  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf40, (2048, 49, 32), (1568, 32, 1), 0), out=buf41)
        del buf39
        buf42 = reinterpret_tensor(buf40, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf40  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf41, buf42, 3211264, grid=grid(3211264), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (25088, 128), (128, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (25088, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 128), (1, 128), 0), out=buf43)
        del arg44_1
        buf47 = reinterpret_tensor(buf42, (8, 3136, 128), (401408, 128, 1), 0); del buf42  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_13.run(buf27, buf43, arg45_1, arg46_1, arg47_1, buf47, 25088, 128, grid=grid(25088), stream=stream0)
        del arg46_1
        del arg47_1
        buf48 = reinterpret_tensor(buf25, (25088, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf47, (25088, 128), (128, 1), 0), reinterpret_tensor(arg48_1, (128, 512), (1, 128), 0), out=buf48)
        del arg48_1
        buf49 = reinterpret_tensor(buf48, (8, 3136, 512), (1605632, 512, 1), 0); del buf48  # reuse
        # Source Nodes: [x_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_9.run(buf49, arg49_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg49_1
        buf50 = reinterpret_tensor(buf47, (25088, 128), (128, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (25088, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 128), (1, 512), 0), out=buf50)
        del arg50_1
        del buf49
        buf51 = reinterpret_tensor(buf50, (8, 28, 28, 2, 2, 128), (401408, 14336, 256, 128, 7168, 1), 0); del buf50  # reuse
        # Source Nodes: [x_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf51, buf27, buf43, arg45_1, arg51_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg45_1
        del arg51_1
        del buf27
        buf55 = reinterpret_tensor(buf43, (8, 28, 28, 512), (401408, 14336, 512, 1), 0); del buf43  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_15.run(buf51, arg52_1, arg53_1, buf55, 6272, 512, grid=grid(6272), stream=stream0)
        del arg52_1
        del arg53_1
        del buf51
        buf56 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (6272, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 256), (1, 512), 0), out=buf56)
        del arg54_1
        buf60 = empty((8, 4, 4, 7, 7, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_8, shifted_x_8], Original ATen: [aten.clone, aten.native_layer_norm]
        triton_per_fused_clone_native_layer_norm_16.run(buf56, arg55_1, arg56_1, buf60, 6272, 256, grid=grid(6272), stream=stream0)
        del arg55_1
        del arg56_1
        buf61 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (6272, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 768), (1, 256), 0), out=buf61)
        del arg57_1
        buf62 = reinterpret_tensor(buf60, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf60  # reuse
        # Source Nodes: [attn_10, q_5], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf61, arg58_1, buf62, 1605632, grid=grid(1605632), stream=stream0)
        buf63 = empty((128, 8, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf61, arg58_1, buf63, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf64 = empty((1024, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf63, (1024, 32, 49), (1568, 49, 1), 0), out=buf64)
        buf67 = empty((128, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_11, attn_12], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_19.run(buf64, arg332_1, arg2_1, buf67, 50176, 49, grid=grid(50176), stream=stream0)
        del arg2_1
        del arg332_1
        buf68 = reinterpret_tensor(buf63, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf63  # reuse
        # Source Nodes: [x_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf61, arg58_1, buf68, 1605632, grid=grid(1605632), stream=stream0)
        del arg58_1
        buf69 = reinterpret_tensor(buf62, (1024, 49, 32), (1568, 32, 1), 0); del buf62  # reuse
        # Source Nodes: [x_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf68, (1024, 49, 32), (1568, 32, 1), 0), out=buf69)
        buf70 = reinterpret_tensor(buf68, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf68  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf69, buf70, 1605632, grid=grid(1605632), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (6272, 256), (256, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (6272, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 256), (1, 256), 0), out=buf71)
        del arg59_1
        buf75 = reinterpret_tensor(buf70, (8, 784, 256), (200704, 256, 1), 0); del buf70  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_22.run(buf56, buf71, arg60_1, arg61_1, arg62_1, buf75, 6272, 256, grid=grid(6272), stream=stream0)
        del arg61_1
        del arg62_1
        buf76 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (6272, 256), (256, 1), 0), reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), out=buf76)
        del arg63_1
        buf77 = reinterpret_tensor(buf76, (8, 784, 1024), (802816, 1024, 1), 0); del buf76  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf77, arg64_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg64_1
        buf78 = reinterpret_tensor(buf75, (6272, 256), (256, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 256), (1, 1024), 0), out=buf78)
        del arg65_1
        buf79 = reinterpret_tensor(buf78, (8, 784, 256), (200704, 256, 1), 0); del buf78  # reuse
        buf80 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf81 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm1, x_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_24.run(buf79, buf56, buf71, arg60_1, arg66_1, buf80, buf81, 6272, 256, grid=grid(6272), stream=stream0)
        del arg60_1
        del arg66_1
        buf83 = reinterpret_tensor(buf71, (8, 4, 4, 7, 7, 256), (200704, 50176, 12544, 1792, 256, 1), 0); del buf71  # reuse
        # Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf79, buf80, buf81, arg67_1, arg68_1, buf83, 1605632, grid=grid(1605632), stream=stream0)
        del arg67_1
        del arg68_1
        del buf80
        del buf81
        buf84 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (6272, 256), (256, 1), 0), reinterpret_tensor(arg69_1, (256, 768), (1, 256), 0), out=buf84)
        del arg69_1
        buf85 = reinterpret_tensor(buf83, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf83  # reuse
        # Source Nodes: [attn_14, q_7], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf84, arg70_1, buf85, 1605632, grid=grid(1605632), stream=stream0)
        buf86 = reinterpret_tensor(buf56, (128, 8, 32, 49), (12544, 1568, 49, 1), 0); del buf56  # reuse
        # Source Nodes: [attn_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf84, arg70_1, buf86, 32768, 49, grid=grid(32768, 49), stream=stream0)
        buf87 = reinterpret_tensor(buf67, (1024, 49, 49), (2401, 49, 1), 0); del buf67  # reuse
        # Source Nodes: [attn_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf85, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf86, (1024, 32, 49), (1568, 49, 1), 0), out=buf87)
        buf91 = reinterpret_tensor(buf64, (128, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf64  # reuse
        # Source Nodes: [attn_18], Original ATen: [aten._softmax]
        triton_per_fused__softmax_26.run(buf87, arg334_1, arg3_1, arg333_1, buf91, 50176, 49, grid=grid(50176), stream=stream0)
        del arg333_1
        del arg334_1
        del arg3_1
        del buf87
        buf92 = reinterpret_tensor(buf86, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf86  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf84, arg70_1, buf92, 1605632, grid=grid(1605632), stream=stream0)
        del arg70_1
        del buf84
        buf93 = reinterpret_tensor(buf85, (1024, 49, 32), (1568, 32, 1), 0); del buf85  # reuse
        # Source Nodes: [x_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf92, (1024, 49, 32), (1568, 32, 1), 0), out=buf93)
        del buf91
        buf94 = reinterpret_tensor(buf92, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf92  # reuse
        # Source Nodes: [x_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf93, buf94, 1605632, grid=grid(1605632), stream=stream0)
        buf95 = reinterpret_tensor(buf93, (6272, 256), (256, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (6272, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), out=buf95)
        del arg71_1
        buf99 = reinterpret_tensor(buf94, (8, 784, 256), (200704, 256, 1), 0); del buf94  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_27.run(buf79, buf95, arg72_1, arg73_1, arg74_1, buf99, 6272, 256, grid=grid(6272), stream=stream0)
        del arg73_1
        del arg74_1
        buf100 = reinterpret_tensor(buf77, (6272, 1024), (1024, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (6272, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 1024), (1, 256), 0), out=buf100)
        del arg75_1
        buf101 = reinterpret_tensor(buf100, (8, 784, 1024), (802816, 1024, 1), 0); del buf100  # reuse
        # Source Nodes: [x_76], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_23.run(buf101, arg76_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg76_1
        buf102 = reinterpret_tensor(buf99, (6272, 256), (256, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 256), (1, 1024), 0), out=buf102)
        del arg77_1
        del buf101
        buf103 = reinterpret_tensor(buf102, (8, 14, 14, 2, 2, 256), (200704, 14336, 512, 256, 7168, 1), 0); del buf102  # reuse
        # Source Nodes: [x_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf103, buf79, buf95, arg72_1, arg78_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg72_1
        del arg78_1
        del buf79
        buf107 = reinterpret_tensor(buf95, (8, 14, 14, 1024), (200704, 14336, 1024, 1), 0); del buf95  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_29.run(buf103, arg79_1, arg80_1, buf107, 1568, 1024, grid=grid(1568), stream=stream0)
        del arg79_1
        del arg80_1
        del buf103
        buf108 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1568, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 512), (1, 1024), 0), out=buf108)
        del arg81_1
        buf112 = empty((8, 2, 2, 7, 7, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_16, shifted_x_16], Original ATen: [aten.clone, aten.native_layer_norm]
        triton_per_fused_clone_native_layer_norm_30.run(buf108, arg82_1, arg83_1, buf112, 1568, 512, grid=grid(1568), stream=stream0)
        del arg82_1
        del arg83_1
        buf113 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (1568, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 1536), (1, 512), 0), out=buf113)
        del arg84_1
        buf114 = reinterpret_tensor(buf112, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf112  # reuse
        # Source Nodes: [attn_20, q_9], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf113, arg85_1, buf114, 802816, grid=grid(802816), stream=stream0)
        buf115 = empty((32, 16, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf113, arg85_1, buf115, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf116 = empty((512, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf115, (512, 32, 49), (1568, 49, 1), 0), out=buf116)
        buf119 = empty((32, 16, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf116, arg335_1, arg4_1, buf119, 25088, 49, grid=grid(25088), stream=stream0)
        del arg335_1
        del arg4_1
        buf120 = reinterpret_tensor(buf115, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf115  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf113, arg85_1, buf120, 802816, grid=grid(802816), stream=stream0)
        del arg85_1
        buf121 = reinterpret_tensor(buf114, (512, 49, 32), (1568, 32, 1), 0); del buf114  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf120, (512, 49, 32), (1568, 32, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf120  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf121, buf122, 802816, grid=grid(802816), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (1568, 512), (512, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (1568, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), out=buf123)
        del arg86_1
        buf127 = reinterpret_tensor(buf122, (8, 196, 512), (100352, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf108, buf123, arg87_1, arg88_1, arg89_1, buf127, 1568, 512, grid=grid(1568), stream=stream0)
        del arg88_1
        del arg89_1
        buf128 = reinterpret_tensor(buf55, (1568, 2048), (2048, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (1568, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 2048), (1, 512), 0), out=buf128)
        del arg90_1
        buf129 = reinterpret_tensor(buf128, (8, 196, 2048), (401408, 2048, 1), 0); del buf128  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf129, arg91_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg91_1
        buf130 = reinterpret_tensor(buf127, (1568, 512), (512, 1), 0); del buf127  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 512), (1, 2048), 0), out=buf130)
        del arg92_1
        buf131 = reinterpret_tensor(buf130, (8, 196, 512), (100352, 512, 1), 0); del buf130  # reuse
        buf132 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf133 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm1, x_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf131, buf108, buf123, arg87_1, arg93_1, buf132, buf133, 1568, 512, grid=grid(1568), stream=stream0)
        del arg87_1
        del arg93_1
        buf135 = reinterpret_tensor(buf123, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf123  # reuse
        # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf131, buf132, buf133, arg94_1, arg95_1, buf135, 802816, grid=grid(802816), stream=stream0)
        del arg94_1
        del arg95_1
        buf136 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (1568, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 1536), (1, 512), 0), out=buf136)
        del arg96_1
        buf137 = reinterpret_tensor(buf135, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf135  # reuse
        # Source Nodes: [attn_24, q_11], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf136, arg97_1, buf137, 802816, grid=grid(802816), stream=stream0)
        buf138 = reinterpret_tensor(buf108, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf108  # reuse
        # Source Nodes: [attn_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf136, arg97_1, buf138, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf139 = reinterpret_tensor(buf119, (512, 49, 49), (2401, 49, 1), 0); del buf119  # reuse
        # Source Nodes: [attn_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf138, (512, 32, 49), (1568, 49, 1), 0), out=buf139)
        buf143 = reinterpret_tensor(buf116, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf116  # reuse
        # Source Nodes: [attn_28], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf139, arg337_1, arg5_1, arg336_1, buf143, 25088, 49, grid=grid(25088), stream=stream0)
        del arg336_1
        del arg337_1
        del arg5_1
        buf144 = reinterpret_tensor(buf138, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf138  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf136, arg97_1, buf144, 802816, grid=grid(802816), stream=stream0)
        del arg97_1
        buf145 = reinterpret_tensor(buf137, (512, 49, 32), (1568, 32, 1), 0); del buf137  # reuse
        # Source Nodes: [x_107], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf144, (512, 49, 32), (1568, 32, 1), 0), out=buf145)
        buf146 = reinterpret_tensor(buf144, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf144  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf145, buf146, 802816, grid=grid(802816), stream=stream0)
        buf147 = reinterpret_tensor(buf145, (1568, 512), (512, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (1568, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), out=buf147)
        del arg98_1
        buf151 = reinterpret_tensor(buf146, (8, 196, 512), (100352, 512, 1), 0); del buf146  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf131, buf147, arg99_1, arg100_1, arg101_1, buf151, 1568, 512, grid=grid(1568), stream=stream0)
        del arg100_1
        del arg101_1
        buf152 = reinterpret_tensor(buf129, (1568, 2048), (2048, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1568, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 2048), (1, 512), 0), out=buf152)
        del arg102_1
        buf153 = reinterpret_tensor(buf152, (8, 196, 2048), (401408, 2048, 1), 0); del buf152  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf153, arg103_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg103_1
        buf154 = reinterpret_tensor(buf151, (1568, 512), (512, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg104_1, (2048, 512), (1, 2048), 0), out=buf154)
        del arg104_1
        buf155 = reinterpret_tensor(buf154, (8, 196, 512), (100352, 512, 1), 0); del buf154  # reuse
        buf159 = empty((8, 2, 2, 7, 7, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [contiguous_24, shifted_x_24, x_122], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf155, buf131, buf147, arg99_1, arg105_1, arg106_1, arg107_1, buf159, 1568, 512, grid=grid(1568), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del arg99_1
        buf160 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1568, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 1536), (1, 512), 0), out=buf160)
        del arg108_1
        buf161 = reinterpret_tensor(buf159, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf159  # reuse
        # Source Nodes: [attn_30, q_13], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf160, arg109_1, buf161, 802816, grid=grid(802816), stream=stream0)
        buf162 = reinterpret_tensor(buf147, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf147  # reuse
        # Source Nodes: [attn_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf160, arg109_1, buf162, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf163 = reinterpret_tensor(buf143, (512, 49, 49), (2401, 49, 1), 0); del buf143  # reuse
        # Source Nodes: [attn_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf162, (512, 32, 49), (1568, 49, 1), 0), out=buf163)
        buf166 = reinterpret_tensor(buf139, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf139  # reuse
        # Source Nodes: [attn_31, attn_32], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf163, arg338_1, arg6_1, buf166, 25088, 49, grid=grid(25088), stream=stream0)
        del arg338_1
        del arg6_1
        buf167 = reinterpret_tensor(buf162, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf162  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf160, arg109_1, buf167, 802816, grid=grid(802816), stream=stream0)
        del arg109_1
        buf168 = reinterpret_tensor(buf161, (512, 49, 32), (1568, 32, 1), 0); del buf161  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf167, (512, 49, 32), (1568, 32, 1), 0), out=buf168)
        buf169 = reinterpret_tensor(buf167, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf167  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf168, buf169, 802816, grid=grid(802816), stream=stream0)
        buf170 = reinterpret_tensor(buf168, (1568, 512), (512, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (1568, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 512), (1, 512), 0), out=buf170)
        del arg110_1
        buf174 = reinterpret_tensor(buf169, (8, 196, 512), (100352, 512, 1), 0); del buf169  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf155, buf170, arg111_1, arg112_1, arg113_1, buf174, 1568, 512, grid=grid(1568), stream=stream0)
        del arg112_1
        del arg113_1
        buf175 = reinterpret_tensor(buf153, (1568, 2048), (2048, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1568, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 2048), (1, 512), 0), out=buf175)
        del arg114_1
        buf176 = reinterpret_tensor(buf175, (8, 196, 2048), (401408, 2048, 1), 0); del buf175  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf176, arg115_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg115_1
        buf177 = reinterpret_tensor(buf174, (1568, 512), (512, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg116_1, (2048, 512), (1, 2048), 0), out=buf177)
        del arg116_1
        buf178 = reinterpret_tensor(buf177, (8, 196, 512), (100352, 512, 1), 0); del buf177  # reuse
        buf179 = buf133; del buf133  # reuse
        buf180 = buf132; del buf132  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___norm1, x_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf178, buf155, buf170, arg111_1, arg117_1, buf179, buf180, 1568, 512, grid=grid(1568), stream=stream0)
        del arg111_1
        del arg117_1
        buf182 = reinterpret_tensor(buf170, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf170  # reuse
        # Source Nodes: [contiguous_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf178, buf179, buf180, arg118_1, arg119_1, buf182, 802816, grid=grid(802816), stream=stream0)
        del arg118_1
        del arg119_1
        buf183 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1568, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 1536), (1, 512), 0), out=buf183)
        del arg120_1
        buf184 = reinterpret_tensor(buf182, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf182  # reuse
        # Source Nodes: [attn_34, q_15], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf183, arg121_1, buf184, 802816, grid=grid(802816), stream=stream0)
        buf185 = reinterpret_tensor(buf155, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf155  # reuse
        # Source Nodes: [attn_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf183, arg121_1, buf185, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf186 = reinterpret_tensor(buf166, (512, 49, 49), (2401, 49, 1), 0); del buf166  # reuse
        # Source Nodes: [attn_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf185, (512, 32, 49), (1568, 49, 1), 0), out=buf186)
        buf190 = reinterpret_tensor(buf163, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf163  # reuse
        # Source Nodes: [attn_38], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf186, arg340_1, arg7_1, arg339_1, buf190, 25088, 49, grid=grid(25088), stream=stream0)
        del arg339_1
        del arg340_1
        del arg7_1
        buf191 = reinterpret_tensor(buf185, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf185  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf183, arg121_1, buf191, 802816, grid=grid(802816), stream=stream0)
        del arg121_1
        buf192 = reinterpret_tensor(buf184, (512, 49, 32), (1568, 32, 1), 0); del buf184  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf191, (512, 49, 32), (1568, 32, 1), 0), out=buf192)
        buf193 = reinterpret_tensor(buf191, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf191  # reuse
        # Source Nodes: [x_144], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf192, buf193, 802816, grid=grid(802816), stream=stream0)
        buf194 = reinterpret_tensor(buf192, (1568, 512), (512, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), out=buf194)
        del arg122_1
        buf198 = reinterpret_tensor(buf193, (8, 196, 512), (100352, 512, 1), 0); del buf193  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf178, buf194, arg123_1, arg124_1, arg125_1, buf198, 1568, 512, grid=grid(1568), stream=stream0)
        del arg124_1
        del arg125_1
        buf199 = reinterpret_tensor(buf176, (1568, 2048), (2048, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (1568, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 2048), (1, 512), 0), out=buf199)
        del arg126_1
        buf200 = reinterpret_tensor(buf199, (8, 196, 2048), (401408, 2048, 1), 0); del buf199  # reuse
        # Source Nodes: [x_153], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf200, arg127_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg127_1
        buf201 = reinterpret_tensor(buf198, (1568, 512), (512, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg128_1, (2048, 512), (1, 2048), 0), out=buf201)
        del arg128_1
        buf202 = reinterpret_tensor(buf201, (8, 196, 512), (100352, 512, 1), 0); del buf201  # reuse
        buf206 = reinterpret_tensor(buf131, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf131  # reuse
        # Source Nodes: [contiguous_32, shifted_x_32, x_158], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf202, buf178, buf194, arg123_1, arg129_1, arg130_1, arg131_1, buf206, 1568, 512, grid=grid(1568), stream=stream0)
        del arg123_1
        del arg129_1
        del arg130_1
        del arg131_1
        buf207 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (1568, 512), (512, 1), 0), reinterpret_tensor(arg132_1, (512, 1536), (1, 512), 0), out=buf207)
        del arg132_1
        buf208 = reinterpret_tensor(buf206, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf206  # reuse
        # Source Nodes: [attn_40, q_17], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf207, arg133_1, buf208, 802816, grid=grid(802816), stream=stream0)
        buf209 = reinterpret_tensor(buf194, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf194  # reuse
        # Source Nodes: [attn_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf207, arg133_1, buf209, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf210 = reinterpret_tensor(buf190, (512, 49, 49), (2401, 49, 1), 0); del buf190  # reuse
        # Source Nodes: [attn_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf209, (512, 32, 49), (1568, 49, 1), 0), out=buf210)
        buf213 = reinterpret_tensor(buf186, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf186  # reuse
        # Source Nodes: [attn_41, attn_42], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf210, arg341_1, arg8_1, buf213, 25088, 49, grid=grid(25088), stream=stream0)
        del arg341_1
        del arg8_1
        buf214 = reinterpret_tensor(buf209, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf209  # reuse
        # Source Nodes: [x_161], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf207, arg133_1, buf214, 802816, grid=grid(802816), stream=stream0)
        del arg133_1
        buf215 = reinterpret_tensor(buf208, (512, 49, 32), (1568, 32, 1), 0); del buf208  # reuse
        # Source Nodes: [x_161], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf214, (512, 49, 32), (1568, 32, 1), 0), out=buf215)
        buf216 = reinterpret_tensor(buf214, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf214  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf215, buf216, 802816, grid=grid(802816), stream=stream0)
        buf217 = reinterpret_tensor(buf215, (1568, 512), (512, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (1568, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 512), (1, 512), 0), out=buf217)
        del arg134_1
        buf221 = reinterpret_tensor(buf216, (8, 196, 512), (100352, 512, 1), 0); del buf216  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___4___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf202, buf217, arg135_1, arg136_1, arg137_1, buf221, 1568, 512, grid=grid(1568), stream=stream0)
        del arg136_1
        del arg137_1
        buf222 = reinterpret_tensor(buf200, (1568, 2048), (2048, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (1568, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 2048), (1, 512), 0), out=buf222)
        del arg138_1
        buf223 = reinterpret_tensor(buf222, (8, 196, 2048), (401408, 2048, 1), 0); del buf222  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf223, arg139_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg139_1
        buf224 = reinterpret_tensor(buf221, (1568, 512), (512, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 512), (1, 2048), 0), out=buf224)
        del arg140_1
        buf225 = reinterpret_tensor(buf224, (8, 196, 512), (100352, 512, 1), 0); del buf224  # reuse
        buf226 = buf180; del buf180  # reuse
        buf227 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___norm1, x_176], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf225, buf202, buf217, arg135_1, arg141_1, buf226, buf227, 1568, 512, grid=grid(1568), stream=stream0)
        del arg135_1
        del arg141_1
        buf229 = reinterpret_tensor(buf217, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf217  # reuse
        # Source Nodes: [contiguous_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf225, buf226, buf227, arg142_1, arg143_1, buf229, 802816, grid=grid(802816), stream=stream0)
        del arg142_1
        del arg143_1
        buf230 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf229, (1568, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 1536), (1, 512), 0), out=buf230)
        del arg144_1
        buf231 = reinterpret_tensor(buf229, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf229  # reuse
        # Source Nodes: [attn_44, q_19], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf230, arg145_1, buf231, 802816, grid=grid(802816), stream=stream0)
        buf232 = reinterpret_tensor(buf202, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf202  # reuse
        # Source Nodes: [attn_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf230, arg145_1, buf232, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf233 = reinterpret_tensor(buf213, (512, 49, 49), (2401, 49, 1), 0); del buf213  # reuse
        # Source Nodes: [attn_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf232, (512, 32, 49), (1568, 49, 1), 0), out=buf233)
        buf237 = reinterpret_tensor(buf210, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf210  # reuse
        # Source Nodes: [attn_48], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf233, arg343_1, arg9_1, arg342_1, buf237, 25088, 49, grid=grid(25088), stream=stream0)
        del arg342_1
        del arg343_1
        del arg9_1
        buf238 = reinterpret_tensor(buf232, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf232  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf230, arg145_1, buf238, 802816, grid=grid(802816), stream=stream0)
        del arg145_1
        buf239 = reinterpret_tensor(buf231, (512, 49, 32), (1568, 32, 1), 0); del buf231  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf238, (512, 49, 32), (1568, 32, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf238, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf238  # reuse
        # Source Nodes: [x_180], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf239, buf240, 802816, grid=grid(802816), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (1568, 512), (512, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (1568, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 512), (1, 512), 0), out=buf241)
        del arg146_1
        buf245 = reinterpret_tensor(buf240, (8, 196, 512), (100352, 512, 1), 0); del buf240  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf225, buf241, arg147_1, arg148_1, arg149_1, buf245, 1568, 512, grid=grid(1568), stream=stream0)
        del arg148_1
        del arg149_1
        buf246 = reinterpret_tensor(buf223, (1568, 2048), (2048, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (1568, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 2048), (1, 512), 0), out=buf246)
        del arg150_1
        buf247 = reinterpret_tensor(buf246, (8, 196, 2048), (401408, 2048, 1), 0); del buf246  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf247, arg151_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg151_1
        buf248 = reinterpret_tensor(buf245, (1568, 512), (512, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg152_1, (2048, 512), (1, 2048), 0), out=buf248)
        del arg152_1
        buf249 = reinterpret_tensor(buf248, (8, 196, 512), (100352, 512, 1), 0); del buf248  # reuse
        buf253 = reinterpret_tensor(buf178, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf178  # reuse
        # Source Nodes: [contiguous_40, shifted_x_40, x_194], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf249, buf225, buf241, arg147_1, arg153_1, arg154_1, arg155_1, buf253, 1568, 512, grid=grid(1568), stream=stream0)
        del arg147_1
        del arg153_1
        del arg154_1
        del arg155_1
        buf254 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (1568, 512), (512, 1), 0), reinterpret_tensor(arg156_1, (512, 1536), (1, 512), 0), out=buf254)
        del arg156_1
        buf255 = reinterpret_tensor(buf253, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf253  # reuse
        # Source Nodes: [attn_50, q_21], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf254, arg157_1, buf255, 802816, grid=grid(802816), stream=stream0)
        buf256 = reinterpret_tensor(buf241, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf241  # reuse
        # Source Nodes: [attn_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf254, arg157_1, buf256, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf257 = reinterpret_tensor(buf237, (512, 49, 49), (2401, 49, 1), 0); del buf237  # reuse
        # Source Nodes: [attn_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf256, (512, 32, 49), (1568, 49, 1), 0), out=buf257)
        buf260 = reinterpret_tensor(buf233, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf233  # reuse
        # Source Nodes: [attn_51, attn_52], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf257, arg344_1, arg10_1, buf260, 25088, 49, grid=grid(25088), stream=stream0)
        del arg10_1
        del arg344_1
        buf261 = reinterpret_tensor(buf256, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf256  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf254, arg157_1, buf261, 802816, grid=grid(802816), stream=stream0)
        del arg157_1
        buf262 = reinterpret_tensor(buf255, (512, 49, 32), (1568, 32, 1), 0); del buf255  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf261, (512, 49, 32), (1568, 32, 1), 0), out=buf262)
        buf263 = reinterpret_tensor(buf261, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf261  # reuse
        # Source Nodes: [x_198], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf262, buf263, 802816, grid=grid(802816), stream=stream0)
        buf264 = reinterpret_tensor(buf262, (1568, 512), (512, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (1568, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 512), (1, 512), 0), out=buf264)
        del arg158_1
        buf268 = reinterpret_tensor(buf263, (8, 196, 512), (100352, 512, 1), 0); del buf263  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___6___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf249, buf264, arg159_1, arg160_1, arg161_1, buf268, 1568, 512, grid=grid(1568), stream=stream0)
        del arg160_1
        del arg161_1
        buf269 = reinterpret_tensor(buf247, (1568, 2048), (2048, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (1568, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 2048), (1, 512), 0), out=buf269)
        del arg162_1
        buf270 = reinterpret_tensor(buf269, (8, 196, 2048), (401408, 2048, 1), 0); del buf269  # reuse
        # Source Nodes: [x_207], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf270, arg163_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg163_1
        buf271 = reinterpret_tensor(buf268, (1568, 512), (512, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg164_1, (2048, 512), (1, 2048), 0), out=buf271)
        del arg164_1
        buf272 = reinterpret_tensor(buf271, (8, 196, 512), (100352, 512, 1), 0); del buf271  # reuse
        buf273 = buf227; del buf227  # reuse
        buf274 = buf226; del buf226  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___norm1, x_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf272, buf249, buf264, arg159_1, arg165_1, buf273, buf274, 1568, 512, grid=grid(1568), stream=stream0)
        del arg159_1
        del arg165_1
        buf276 = reinterpret_tensor(buf264, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf264  # reuse
        # Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf272, buf273, buf274, arg166_1, arg167_1, buf276, 802816, grid=grid(802816), stream=stream0)
        del arg166_1
        del arg167_1
        buf277 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf276, (1568, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 1536), (1, 512), 0), out=buf277)
        del arg168_1
        buf278 = reinterpret_tensor(buf276, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf276  # reuse
        # Source Nodes: [attn_54, q_23], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf277, arg169_1, buf278, 802816, grid=grid(802816), stream=stream0)
        buf279 = reinterpret_tensor(buf249, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf249  # reuse
        # Source Nodes: [attn_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf277, arg169_1, buf279, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf280 = reinterpret_tensor(buf260, (512, 49, 49), (2401, 49, 1), 0); del buf260  # reuse
        # Source Nodes: [attn_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf279, (512, 32, 49), (1568, 49, 1), 0), out=buf280)
        buf284 = reinterpret_tensor(buf257, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf257  # reuse
        # Source Nodes: [attn_58], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf280, arg346_1, arg11_1, arg345_1, buf284, 25088, 49, grid=grid(25088), stream=stream0)
        del arg11_1
        del arg345_1
        del arg346_1
        buf285 = reinterpret_tensor(buf279, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf279  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf277, arg169_1, buf285, 802816, grid=grid(802816), stream=stream0)
        del arg169_1
        buf286 = reinterpret_tensor(buf278, (512, 49, 32), (1568, 32, 1), 0); del buf278  # reuse
        # Source Nodes: [x_215], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf285, (512, 49, 32), (1568, 32, 1), 0), out=buf286)
        buf287 = reinterpret_tensor(buf285, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf285  # reuse
        # Source Nodes: [x_216], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf286, buf287, 802816, grid=grid(802816), stream=stream0)
        buf288 = reinterpret_tensor(buf286, (1568, 512), (512, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf287, (1568, 512), (512, 1), 0), reinterpret_tensor(arg170_1, (512, 512), (1, 512), 0), out=buf288)
        del arg170_1
        buf292 = reinterpret_tensor(buf287, (8, 196, 512), (100352, 512, 1), 0); del buf287  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf272, buf288, arg171_1, arg172_1, arg173_1, buf292, 1568, 512, grid=grid(1568), stream=stream0)
        del arg172_1
        del arg173_1
        buf293 = reinterpret_tensor(buf270, (1568, 2048), (2048, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (1568, 512), (512, 1), 0), reinterpret_tensor(arg174_1, (512, 2048), (1, 512), 0), out=buf293)
        del arg174_1
        buf294 = reinterpret_tensor(buf293, (8, 196, 2048), (401408, 2048, 1), 0); del buf293  # reuse
        # Source Nodes: [x_225], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf294, arg175_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg175_1
        buf295 = reinterpret_tensor(buf292, (1568, 512), (512, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 512), (1, 2048), 0), out=buf295)
        del arg176_1
        buf296 = reinterpret_tensor(buf295, (8, 196, 512), (100352, 512, 1), 0); del buf295  # reuse
        buf300 = reinterpret_tensor(buf225, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf225  # reuse
        # Source Nodes: [contiguous_48, shifted_x_48, x_230], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf296, buf272, buf288, arg171_1, arg177_1, arg178_1, arg179_1, buf300, 1568, 512, grid=grid(1568), stream=stream0)
        del arg171_1
        del arg177_1
        del arg178_1
        del arg179_1
        buf301 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (1568, 512), (512, 1), 0), reinterpret_tensor(arg180_1, (512, 1536), (1, 512), 0), out=buf301)
        del arg180_1
        buf302 = reinterpret_tensor(buf300, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf300  # reuse
        # Source Nodes: [attn_60, q_25], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf301, arg181_1, buf302, 802816, grid=grid(802816), stream=stream0)
        buf303 = reinterpret_tensor(buf288, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf288  # reuse
        # Source Nodes: [attn_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf301, arg181_1, buf303, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf304 = reinterpret_tensor(buf284, (512, 49, 49), (2401, 49, 1), 0); del buf284  # reuse
        # Source Nodes: [attn_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf303, (512, 32, 49), (1568, 49, 1), 0), out=buf304)
        buf307 = reinterpret_tensor(buf280, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf280  # reuse
        # Source Nodes: [attn_61, attn_62], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf304, arg347_1, arg12_1, buf307, 25088, 49, grid=grid(25088), stream=stream0)
        del arg12_1
        del arg347_1
        buf308 = reinterpret_tensor(buf303, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf303  # reuse
        # Source Nodes: [x_233], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf301, arg181_1, buf308, 802816, grid=grid(802816), stream=stream0)
        del arg181_1
        buf309 = reinterpret_tensor(buf302, (512, 49, 32), (1568, 32, 1), 0); del buf302  # reuse
        # Source Nodes: [x_233], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf308, (512, 49, 32), (1568, 32, 1), 0), out=buf309)
        buf310 = reinterpret_tensor(buf308, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf308  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf309, buf310, 802816, grid=grid(802816), stream=stream0)
        buf311 = reinterpret_tensor(buf309, (1568, 512), (512, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 512), (1, 512), 0), out=buf311)
        del arg182_1
        buf315 = reinterpret_tensor(buf310, (8, 196, 512), (100352, 512, 1), 0); del buf310  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___8___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf296, buf311, arg183_1, arg184_1, arg185_1, buf315, 1568, 512, grid=grid(1568), stream=stream0)
        del arg184_1
        del arg185_1
        buf316 = reinterpret_tensor(buf294, (1568, 2048), (2048, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (1568, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 2048), (1, 512), 0), out=buf316)
        del arg186_1
        buf317 = reinterpret_tensor(buf316, (8, 196, 2048), (401408, 2048, 1), 0); del buf316  # reuse
        # Source Nodes: [x_243], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf317, arg187_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        buf318 = reinterpret_tensor(buf315, (1568, 512), (512, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 512), (1, 2048), 0), out=buf318)
        del arg188_1
        buf319 = reinterpret_tensor(buf318, (8, 196, 512), (100352, 512, 1), 0); del buf318  # reuse
        buf320 = buf274; del buf274  # reuse
        buf321 = buf273; del buf273  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___norm1, x_248], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf319, buf296, buf311, arg183_1, arg189_1, buf320, buf321, 1568, 512, grid=grid(1568), stream=stream0)
        del arg183_1
        del arg189_1
        buf323 = reinterpret_tensor(buf311, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf311  # reuse
        # Source Nodes: [contiguous_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf319, buf320, buf321, arg190_1, arg191_1, buf323, 802816, grid=grid(802816), stream=stream0)
        del arg190_1
        del arg191_1
        buf324 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (1568, 512), (512, 1), 0), reinterpret_tensor(arg192_1, (512, 1536), (1, 512), 0), out=buf324)
        del arg192_1
        buf325 = reinterpret_tensor(buf323, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf323  # reuse
        # Source Nodes: [attn_64, q_27], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf324, arg193_1, buf325, 802816, grid=grid(802816), stream=stream0)
        buf326 = reinterpret_tensor(buf296, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf296  # reuse
        # Source Nodes: [attn_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf324, arg193_1, buf326, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf327 = reinterpret_tensor(buf307, (512, 49, 49), (2401, 49, 1), 0); del buf307  # reuse
        # Source Nodes: [attn_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf326, (512, 32, 49), (1568, 49, 1), 0), out=buf327)
        buf331 = reinterpret_tensor(buf304, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf304  # reuse
        # Source Nodes: [attn_68], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf327, arg349_1, arg13_1, arg348_1, buf331, 25088, 49, grid=grid(25088), stream=stream0)
        del arg13_1
        del arg348_1
        del arg349_1
        buf332 = reinterpret_tensor(buf326, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf326  # reuse
        # Source Nodes: [x_251], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf324, arg193_1, buf332, 802816, grid=grid(802816), stream=stream0)
        del arg193_1
        buf333 = reinterpret_tensor(buf325, (512, 49, 32), (1568, 32, 1), 0); del buf325  # reuse
        # Source Nodes: [x_251], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf332, (512, 49, 32), (1568, 32, 1), 0), out=buf333)
        buf334 = reinterpret_tensor(buf332, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf332  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf333, buf334, 802816, grid=grid(802816), stream=stream0)
        buf335 = reinterpret_tensor(buf333, (1568, 512), (512, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (1568, 512), (512, 1), 0), reinterpret_tensor(arg194_1, (512, 512), (1, 512), 0), out=buf335)
        del arg194_1
        buf339 = reinterpret_tensor(buf334, (8, 196, 512), (100352, 512, 1), 0); del buf334  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf319, buf335, arg195_1, arg196_1, arg197_1, buf339, 1568, 512, grid=grid(1568), stream=stream0)
        del arg196_1
        del arg197_1
        buf340 = reinterpret_tensor(buf317, (1568, 2048), (2048, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (1568, 512), (512, 1), 0), reinterpret_tensor(arg198_1, (512, 2048), (1, 512), 0), out=buf340)
        del arg198_1
        buf341 = reinterpret_tensor(buf340, (8, 196, 2048), (401408, 2048, 1), 0); del buf340  # reuse
        # Source Nodes: [x_261], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf341, arg199_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg199_1
        buf342 = reinterpret_tensor(buf339, (1568, 512), (512, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 512), (1, 2048), 0), out=buf342)
        del arg200_1
        buf343 = reinterpret_tensor(buf342, (8, 196, 512), (100352, 512, 1), 0); del buf342  # reuse
        buf347 = reinterpret_tensor(buf272, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf272  # reuse
        # Source Nodes: [contiguous_56, shifted_x_56, x_266], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf343, buf319, buf335, arg195_1, arg201_1, arg202_1, arg203_1, buf347, 1568, 512, grid=grid(1568), stream=stream0)
        del arg195_1
        del arg201_1
        del arg202_1
        del arg203_1
        buf348 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf347, (1568, 512), (512, 1), 0), reinterpret_tensor(arg204_1, (512, 1536), (1, 512), 0), out=buf348)
        del arg204_1
        buf349 = reinterpret_tensor(buf347, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf347  # reuse
        # Source Nodes: [attn_70, q_29], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf348, arg205_1, buf349, 802816, grid=grid(802816), stream=stream0)
        buf350 = reinterpret_tensor(buf335, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf335  # reuse
        # Source Nodes: [attn_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf348, arg205_1, buf350, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf351 = reinterpret_tensor(buf331, (512, 49, 49), (2401, 49, 1), 0); del buf331  # reuse
        # Source Nodes: [attn_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf349, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf350, (512, 32, 49), (1568, 49, 1), 0), out=buf351)
        buf354 = reinterpret_tensor(buf327, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf327  # reuse
        # Source Nodes: [attn_71, attn_72], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf351, arg350_1, arg14_1, buf354, 25088, 49, grid=grid(25088), stream=stream0)
        del arg14_1
        del arg350_1
        buf355 = reinterpret_tensor(buf350, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf350  # reuse
        # Source Nodes: [x_269], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf348, arg205_1, buf355, 802816, grid=grid(802816), stream=stream0)
        del arg205_1
        buf356 = reinterpret_tensor(buf349, (512, 49, 32), (1568, 32, 1), 0); del buf349  # reuse
        # Source Nodes: [x_269], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf355, (512, 49, 32), (1568, 32, 1), 0), out=buf356)
        buf357 = reinterpret_tensor(buf355, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf355  # reuse
        # Source Nodes: [x_270], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf356, buf357, 802816, grid=grid(802816), stream=stream0)
        buf358 = reinterpret_tensor(buf356, (1568, 512), (512, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf357, (1568, 512), (512, 1), 0), reinterpret_tensor(arg206_1, (512, 512), (1, 512), 0), out=buf358)
        del arg206_1
        buf362 = reinterpret_tensor(buf357, (8, 196, 512), (100352, 512, 1), 0); del buf357  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___10___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf343, buf358, arg207_1, arg208_1, arg209_1, buf362, 1568, 512, grid=grid(1568), stream=stream0)
        del arg208_1
        del arg209_1
        buf363 = reinterpret_tensor(buf341, (1568, 2048), (2048, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (1568, 512), (512, 1), 0), reinterpret_tensor(arg210_1, (512, 2048), (1, 512), 0), out=buf363)
        del arg210_1
        buf364 = reinterpret_tensor(buf363, (8, 196, 2048), (401408, 2048, 1), 0); del buf363  # reuse
        # Source Nodes: [x_279], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf364, arg211_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg211_1
        buf365 = reinterpret_tensor(buf362, (1568, 512), (512, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 512), (1, 2048), 0), out=buf365)
        del arg212_1
        buf366 = reinterpret_tensor(buf365, (8, 196, 512), (100352, 512, 1), 0); del buf365  # reuse
        buf367 = buf321; del buf321  # reuse
        buf368 = buf320; del buf320  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___norm1, x_284], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf366, buf343, buf358, arg207_1, arg213_1, buf367, buf368, 1568, 512, grid=grid(1568), stream=stream0)
        del arg207_1
        del arg213_1
        buf370 = reinterpret_tensor(buf358, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf358  # reuse
        # Source Nodes: [contiguous_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf366, buf367, buf368, arg214_1, arg215_1, buf370, 802816, grid=grid(802816), stream=stream0)
        del arg214_1
        del arg215_1
        buf371 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (1568, 512), (512, 1), 0), reinterpret_tensor(arg216_1, (512, 1536), (1, 512), 0), out=buf371)
        del arg216_1
        buf372 = reinterpret_tensor(buf370, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf370  # reuse
        # Source Nodes: [attn_74, q_31], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf371, arg217_1, buf372, 802816, grid=grid(802816), stream=stream0)
        buf373 = reinterpret_tensor(buf343, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf343  # reuse
        # Source Nodes: [attn_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf371, arg217_1, buf373, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf374 = reinterpret_tensor(buf354, (512, 49, 49), (2401, 49, 1), 0); del buf354  # reuse
        # Source Nodes: [attn_74], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf373, (512, 32, 49), (1568, 49, 1), 0), out=buf374)
        buf378 = reinterpret_tensor(buf351, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf351  # reuse
        # Source Nodes: [attn_78], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf374, arg352_1, arg15_1, arg351_1, buf378, 25088, 49, grid=grid(25088), stream=stream0)
        del arg15_1
        del arg351_1
        del arg352_1
        buf379 = reinterpret_tensor(buf373, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf373  # reuse
        # Source Nodes: [x_287], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf371, arg217_1, buf379, 802816, grid=grid(802816), stream=stream0)
        del arg217_1
        buf380 = reinterpret_tensor(buf372, (512, 49, 32), (1568, 32, 1), 0); del buf372  # reuse
        # Source Nodes: [x_287], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf379, (512, 49, 32), (1568, 32, 1), 0), out=buf380)
        buf381 = reinterpret_tensor(buf379, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf379  # reuse
        # Source Nodes: [x_288], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf380, buf381, 802816, grid=grid(802816), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (1568, 512), (512, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1568, 512), (512, 1), 0), reinterpret_tensor(arg218_1, (512, 512), (1, 512), 0), out=buf382)
        del arg218_1
        buf386 = reinterpret_tensor(buf381, (8, 196, 512), (100352, 512, 1), 0); del buf381  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf366, buf382, arg219_1, arg220_1, arg221_1, buf386, 1568, 512, grid=grid(1568), stream=stream0)
        del arg220_1
        del arg221_1
        buf387 = reinterpret_tensor(buf364, (1568, 2048), (2048, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf386, (1568, 512), (512, 1), 0), reinterpret_tensor(arg222_1, (512, 2048), (1, 512), 0), out=buf387)
        del arg222_1
        buf388 = reinterpret_tensor(buf387, (8, 196, 2048), (401408, 2048, 1), 0); del buf387  # reuse
        # Source Nodes: [x_297], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf388, arg223_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg223_1
        buf389 = reinterpret_tensor(buf386, (1568, 512), (512, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg224_1, (2048, 512), (1, 2048), 0), out=buf389)
        del arg224_1
        buf390 = reinterpret_tensor(buf389, (8, 196, 512), (100352, 512, 1), 0); del buf389  # reuse
        buf394 = reinterpret_tensor(buf319, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf319  # reuse
        # Source Nodes: [contiguous_64, shifted_x_64, x_302], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf390, buf366, buf382, arg219_1, arg225_1, arg226_1, arg227_1, buf394, 1568, 512, grid=grid(1568), stream=stream0)
        del arg219_1
        del arg225_1
        del arg226_1
        del arg227_1
        buf395 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 512), (512, 1), 0), reinterpret_tensor(arg228_1, (512, 1536), (1, 512), 0), out=buf395)
        del arg228_1
        buf396 = reinterpret_tensor(buf394, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf394  # reuse
        # Source Nodes: [attn_80, q_33], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf395, arg229_1, buf396, 802816, grid=grid(802816), stream=stream0)
        buf397 = reinterpret_tensor(buf382, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf382  # reuse
        # Source Nodes: [attn_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf395, arg229_1, buf397, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf398 = reinterpret_tensor(buf378, (512, 49, 49), (2401, 49, 1), 0); del buf378  # reuse
        # Source Nodes: [attn_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf397, (512, 32, 49), (1568, 49, 1), 0), out=buf398)
        buf401 = reinterpret_tensor(buf374, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf374  # reuse
        # Source Nodes: [attn_81, attn_82], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf398, arg353_1, arg16_1, buf401, 25088, 49, grid=grid(25088), stream=stream0)
        del arg16_1
        del arg353_1
        buf402 = reinterpret_tensor(buf397, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf397  # reuse
        # Source Nodes: [x_305], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf395, arg229_1, buf402, 802816, grid=grid(802816), stream=stream0)
        del arg229_1
        buf403 = reinterpret_tensor(buf396, (512, 49, 32), (1568, 32, 1), 0); del buf396  # reuse
        # Source Nodes: [x_305], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf402, (512, 49, 32), (1568, 32, 1), 0), out=buf403)
        buf404 = reinterpret_tensor(buf402, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf402  # reuse
        # Source Nodes: [x_306], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf403, buf404, 802816, grid=grid(802816), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (1568, 512), (512, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (1568, 512), (512, 1), 0), reinterpret_tensor(arg230_1, (512, 512), (1, 512), 0), out=buf405)
        del arg230_1
        buf409 = reinterpret_tensor(buf404, (8, 196, 512), (100352, 512, 1), 0); del buf404  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___12___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf390, buf405, arg231_1, arg232_1, arg233_1, buf409, 1568, 512, grid=grid(1568), stream=stream0)
        del arg232_1
        del arg233_1
        buf410 = reinterpret_tensor(buf388, (1568, 2048), (2048, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf409, (1568, 512), (512, 1), 0), reinterpret_tensor(arg234_1, (512, 2048), (1, 512), 0), out=buf410)
        del arg234_1
        buf411 = reinterpret_tensor(buf410, (8, 196, 2048), (401408, 2048, 1), 0); del buf410  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf411, arg235_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg235_1
        buf412 = reinterpret_tensor(buf409, (1568, 512), (512, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg236_1, (2048, 512), (1, 2048), 0), out=buf412)
        del arg236_1
        buf413 = reinterpret_tensor(buf412, (8, 196, 512), (100352, 512, 1), 0); del buf412  # reuse
        buf414 = buf368; del buf368  # reuse
        buf415 = buf367; del buf367  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___norm1, x_320], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf413, buf390, buf405, arg231_1, arg237_1, buf414, buf415, 1568, 512, grid=grid(1568), stream=stream0)
        del arg231_1
        del arg237_1
        buf417 = reinterpret_tensor(buf405, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf405  # reuse
        # Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf413, buf414, buf415, arg238_1, arg239_1, buf417, 802816, grid=grid(802816), stream=stream0)
        del arg238_1
        del arg239_1
        buf418 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 512), (512, 1), 0), reinterpret_tensor(arg240_1, (512, 1536), (1, 512), 0), out=buf418)
        del arg240_1
        buf419 = reinterpret_tensor(buf417, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf417  # reuse
        # Source Nodes: [attn_84, q_35], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf418, arg241_1, buf419, 802816, grid=grid(802816), stream=stream0)
        buf420 = reinterpret_tensor(buf390, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf390  # reuse
        # Source Nodes: [attn_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf418, arg241_1, buf420, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf421 = reinterpret_tensor(buf401, (512, 49, 49), (2401, 49, 1), 0); del buf401  # reuse
        # Source Nodes: [attn_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf420, (512, 32, 49), (1568, 49, 1), 0), out=buf421)
        buf425 = reinterpret_tensor(buf398, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf398  # reuse
        # Source Nodes: [attn_88], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf421, arg355_1, arg17_1, arg354_1, buf425, 25088, 49, grid=grid(25088), stream=stream0)
        del arg17_1
        del arg354_1
        del arg355_1
        buf426 = reinterpret_tensor(buf420, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf420  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf418, arg241_1, buf426, 802816, grid=grid(802816), stream=stream0)
        del arg241_1
        buf427 = reinterpret_tensor(buf419, (512, 49, 32), (1568, 32, 1), 0); del buf419  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf425, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf426, (512, 49, 32), (1568, 32, 1), 0), out=buf427)
        buf428 = reinterpret_tensor(buf426, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf426  # reuse
        # Source Nodes: [x_324], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf427, buf428, 802816, grid=grid(802816), stream=stream0)
        buf429 = reinterpret_tensor(buf427, (1568, 512), (512, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (1568, 512), (512, 1), 0), reinterpret_tensor(arg242_1, (512, 512), (1, 512), 0), out=buf429)
        del arg242_1
        buf433 = reinterpret_tensor(buf428, (8, 196, 512), (100352, 512, 1), 0); del buf428  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf413, buf429, arg243_1, arg244_1, arg245_1, buf433, 1568, 512, grid=grid(1568), stream=stream0)
        del arg244_1
        del arg245_1
        buf434 = reinterpret_tensor(buf411, (1568, 2048), (2048, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (1568, 512), (512, 1), 0), reinterpret_tensor(arg246_1, (512, 2048), (1, 512), 0), out=buf434)
        del arg246_1
        buf435 = reinterpret_tensor(buf434, (8, 196, 2048), (401408, 2048, 1), 0); del buf434  # reuse
        # Source Nodes: [x_333], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf435, arg247_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg247_1
        buf436 = reinterpret_tensor(buf433, (1568, 512), (512, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 512), (1, 2048), 0), out=buf436)
        del arg248_1
        buf437 = reinterpret_tensor(buf436, (8, 196, 512), (100352, 512, 1), 0); del buf436  # reuse
        buf441 = reinterpret_tensor(buf366, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf366  # reuse
        # Source Nodes: [contiguous_72, shifted_x_72, x_338], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf437, buf413, buf429, arg243_1, arg249_1, arg250_1, arg251_1, buf441, 1568, 512, grid=grid(1568), stream=stream0)
        del arg243_1
        del arg249_1
        del arg250_1
        del arg251_1
        buf442 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (1568, 512), (512, 1), 0), reinterpret_tensor(arg252_1, (512, 1536), (1, 512), 0), out=buf442)
        del arg252_1
        buf443 = reinterpret_tensor(buf441, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf441  # reuse
        # Source Nodes: [attn_90, q_37], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf442, arg253_1, buf443, 802816, grid=grid(802816), stream=stream0)
        buf444 = reinterpret_tensor(buf429, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf429  # reuse
        # Source Nodes: [attn_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf442, arg253_1, buf444, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf445 = reinterpret_tensor(buf425, (512, 49, 49), (2401, 49, 1), 0); del buf425  # reuse
        # Source Nodes: [attn_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf444, (512, 32, 49), (1568, 49, 1), 0), out=buf445)
        buf448 = reinterpret_tensor(buf421, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf421  # reuse
        # Source Nodes: [attn_91, attn_92], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf445, arg356_1, arg18_1, buf448, 25088, 49, grid=grid(25088), stream=stream0)
        del arg18_1
        del arg356_1
        buf449 = reinterpret_tensor(buf444, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf444  # reuse
        # Source Nodes: [x_341], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf442, arg253_1, buf449, 802816, grid=grid(802816), stream=stream0)
        del arg253_1
        buf450 = reinterpret_tensor(buf443, (512, 49, 32), (1568, 32, 1), 0); del buf443  # reuse
        # Source Nodes: [x_341], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf448, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf449, (512, 49, 32), (1568, 32, 1), 0), out=buf450)
        buf451 = reinterpret_tensor(buf449, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf449  # reuse
        # Source Nodes: [x_342], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf450, buf451, 802816, grid=grid(802816), stream=stream0)
        buf452 = reinterpret_tensor(buf450, (1568, 512), (512, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf451, (1568, 512), (512, 1), 0), reinterpret_tensor(arg254_1, (512, 512), (1, 512), 0), out=buf452)
        del arg254_1
        buf456 = reinterpret_tensor(buf451, (8, 196, 512), (100352, 512, 1), 0); del buf451  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___14___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf437, buf452, arg255_1, arg256_1, arg257_1, buf456, 1568, 512, grid=grid(1568), stream=stream0)
        del arg256_1
        del arg257_1
        buf457 = reinterpret_tensor(buf435, (1568, 2048), (2048, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 512), (512, 1), 0), reinterpret_tensor(arg258_1, (512, 2048), (1, 512), 0), out=buf457)
        del arg258_1
        buf458 = reinterpret_tensor(buf457, (8, 196, 2048), (401408, 2048, 1), 0); del buf457  # reuse
        # Source Nodes: [x_351], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf458, arg259_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg259_1
        buf459 = reinterpret_tensor(buf456, (1568, 512), (512, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg260_1, (2048, 512), (1, 2048), 0), out=buf459)
        del arg260_1
        buf460 = reinterpret_tensor(buf459, (8, 196, 512), (100352, 512, 1), 0); del buf459  # reuse
        buf461 = buf415; del buf415  # reuse
        buf462 = buf414; del buf414  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___norm1, x_356], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf460, buf437, buf452, arg255_1, arg261_1, buf461, buf462, 1568, 512, grid=grid(1568), stream=stream0)
        del arg255_1
        del arg261_1
        buf464 = reinterpret_tensor(buf452, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf452  # reuse
        # Source Nodes: [contiguous_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf460, buf461, buf462, arg262_1, arg263_1, buf464, 802816, grid=grid(802816), stream=stream0)
        del arg262_1
        del arg263_1
        buf465 = buf442; del buf442  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf464, (1568, 512), (512, 1), 0), reinterpret_tensor(arg264_1, (512, 1536), (1, 512), 0), out=buf465)
        del arg264_1
        buf466 = reinterpret_tensor(buf464, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf464  # reuse
        # Source Nodes: [attn_94, q_39], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf465, arg265_1, buf466, 802816, grid=grid(802816), stream=stream0)
        buf467 = reinterpret_tensor(buf437, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf437  # reuse
        # Source Nodes: [attn_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf465, arg265_1, buf467, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf468 = reinterpret_tensor(buf448, (512, 49, 49), (2401, 49, 1), 0); del buf448  # reuse
        # Source Nodes: [attn_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf467, (512, 32, 49), (1568, 49, 1), 0), out=buf468)
        buf472 = reinterpret_tensor(buf445, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf445  # reuse
        # Source Nodes: [attn_98], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf468, arg358_1, arg19_1, arg357_1, buf472, 25088, 49, grid=grid(25088), stream=stream0)
        del arg19_1
        del arg357_1
        del arg358_1
        buf473 = reinterpret_tensor(buf467, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf467  # reuse
        # Source Nodes: [x_359], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf465, arg265_1, buf473, 802816, grid=grid(802816), stream=stream0)
        del arg265_1
        buf474 = reinterpret_tensor(buf466, (512, 49, 32), (1568, 32, 1), 0); del buf466  # reuse
        # Source Nodes: [x_359], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf472, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf473, (512, 49, 32), (1568, 32, 1), 0), out=buf474)
        buf475 = reinterpret_tensor(buf473, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf473  # reuse
        # Source Nodes: [x_360], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf474, buf475, 802816, grid=grid(802816), stream=stream0)
        buf476 = reinterpret_tensor(buf474, (1568, 512), (512, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf475, (1568, 512), (512, 1), 0), reinterpret_tensor(arg266_1, (512, 512), (1, 512), 0), out=buf476)
        del arg266_1
        buf480 = reinterpret_tensor(buf475, (8, 196, 512), (100352, 512, 1), 0); del buf475  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf460, buf476, arg267_1, arg268_1, arg269_1, buf480, 1568, 512, grid=grid(1568), stream=stream0)
        del arg268_1
        del arg269_1
        buf481 = reinterpret_tensor(buf458, (1568, 2048), (2048, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (1568, 512), (512, 1), 0), reinterpret_tensor(arg270_1, (512, 2048), (1, 512), 0), out=buf481)
        del arg270_1
        buf482 = reinterpret_tensor(buf481, (8, 196, 2048), (401408, 2048, 1), 0); del buf481  # reuse
        # Source Nodes: [x_369], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf482, arg271_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg271_1
        buf483 = reinterpret_tensor(buf480, (1568, 512), (512, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf482, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 512), (1, 2048), 0), out=buf483)
        del arg272_1
        buf484 = reinterpret_tensor(buf483, (8, 196, 512), (100352, 512, 1), 0); del buf483  # reuse
        buf488 = reinterpret_tensor(buf413, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf413  # reuse
        # Source Nodes: [contiguous_80, shifted_x_80, x_374], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
        triton_per_fused_add_clone_native_layer_norm_42.run(buf484, buf460, buf476, arg267_1, arg273_1, arg274_1, arg275_1, buf488, 1568, 512, grid=grid(1568), stream=stream0)
        del arg267_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf460
        buf489 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (1568, 512), (512, 1), 0), reinterpret_tensor(arg276_1, (512, 1536), (1, 512), 0), out=buf489)
        del arg276_1
        buf490 = reinterpret_tensor(buf488, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf488  # reuse
        # Source Nodes: [attn_100, q_41], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf489, arg277_1, buf490, 802816, grid=grid(802816), stream=stream0)
        buf491 = reinterpret_tensor(buf476, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf476  # reuse
        # Source Nodes: [attn_100], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf489, arg277_1, buf491, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf492 = reinterpret_tensor(buf472, (512, 49, 49), (2401, 49, 1), 0); del buf472  # reuse
        # Source Nodes: [attn_100], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf491, (512, 32, 49), (1568, 49, 1), 0), out=buf492)
        buf495 = reinterpret_tensor(buf468, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf468  # reuse
        # Source Nodes: [attn_101, attn_102], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_33.run(buf492, arg359_1, arg20_1, buf495, 25088, 49, grid=grid(25088), stream=stream0)
        del arg20_1
        del arg359_1
        buf496 = reinterpret_tensor(buf491, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf491  # reuse
        # Source Nodes: [x_377], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf489, arg277_1, buf496, 802816, grid=grid(802816), stream=stream0)
        del arg277_1
        buf497 = reinterpret_tensor(buf490, (512, 49, 32), (1568, 32, 1), 0); del buf490  # reuse
        # Source Nodes: [x_377], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf496, (512, 49, 32), (1568, 32, 1), 0), out=buf497)
        buf498 = reinterpret_tensor(buf496, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf496  # reuse
        # Source Nodes: [x_378], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf497, buf498, 802816, grid=grid(802816), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (1568, 512), (512, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (1568, 512), (512, 1), 0), reinterpret_tensor(arg278_1, (512, 512), (1, 512), 0), out=buf499)
        del arg278_1
        buf503 = reinterpret_tensor(buf498, (8, 196, 512), (100352, 512, 1), 0); del buf498  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___16___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_36.run(buf484, buf499, arg279_1, arg280_1, arg281_1, buf503, 1568, 512, grid=grid(1568), stream=stream0)
        del arg280_1
        del arg281_1
        buf504 = reinterpret_tensor(buf482, (1568, 2048), (2048, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (1568, 512), (512, 1), 0), reinterpret_tensor(arg282_1, (512, 2048), (1, 512), 0), out=buf504)
        del arg282_1
        buf505 = reinterpret_tensor(buf504, (8, 196, 2048), (401408, 2048, 1), 0); del buf504  # reuse
        # Source Nodes: [x_387], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf505, arg283_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg283_1
        buf506 = reinterpret_tensor(buf503, (1568, 512), (512, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 512), (1, 2048), 0), out=buf506)
        del arg284_1
        buf507 = reinterpret_tensor(buf506, (8, 196, 512), (100352, 512, 1), 0); del buf506  # reuse
        buf508 = buf462; del buf462  # reuse
        buf509 = buf461; del buf461  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___norm1, x_392], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_38.run(buf507, buf484, buf499, arg279_1, arg285_1, buf508, buf509, 1568, 512, grid=grid(1568), stream=stream0)
        del arg279_1
        del arg285_1
        buf511 = reinterpret_tensor(buf499, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf499  # reuse
        # Source Nodes: [contiguous_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf507, buf508, buf509, arg286_1, arg287_1, buf511, 802816, grid=grid(802816), stream=stream0)
        del arg286_1
        del arg287_1
        del buf508
        del buf509
        buf512 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf511, (1568, 512), (512, 1), 0), reinterpret_tensor(arg288_1, (512, 1536), (1, 512), 0), out=buf512)
        del arg288_1
        buf513 = reinterpret_tensor(buf511, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf511  # reuse
        # Source Nodes: [attn_104, q_43], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_31.run(buf512, arg289_1, buf513, 802816, grid=grid(802816), stream=stream0)
        buf514 = reinterpret_tensor(buf484, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf484  # reuse
        # Source Nodes: [attn_104], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf512, arg289_1, buf514, 16384, 49, grid=grid(16384, 49), stream=stream0)
        buf515 = reinterpret_tensor(buf495, (512, 49, 49), (2401, 49, 1), 0); del buf495  # reuse
        # Source Nodes: [attn_104], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf514, (512, 32, 49), (1568, 49, 1), 0), out=buf515)
        buf519 = reinterpret_tensor(buf492, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf492  # reuse
        # Source Nodes: [attn_108], Original ATen: [aten._softmax]
        triton_per_fused__softmax_40.run(buf515, arg361_1, arg21_1, arg360_1, buf519, 25088, 49, grid=grid(25088), stream=stream0)
        del arg21_1
        del arg360_1
        del arg361_1
        del buf515
        buf520 = reinterpret_tensor(buf514, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf514  # reuse
        # Source Nodes: [x_395], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf512, arg289_1, buf520, 802816, grid=grid(802816), stream=stream0)
        del arg289_1
        del buf512
        buf521 = reinterpret_tensor(buf513, (512, 49, 32), (1568, 32, 1), 0); del buf513  # reuse
        # Source Nodes: [x_395], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf519, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf520, (512, 49, 32), (1568, 32, 1), 0), out=buf521)
        del buf519
        buf522 = reinterpret_tensor(buf520, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf520  # reuse
        # Source Nodes: [x_396], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf521, buf522, 802816, grid=grid(802816), stream=stream0)
        buf523 = reinterpret_tensor(buf521, (1568, 512), (512, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf522, (1568, 512), (512, 1), 0), reinterpret_tensor(arg290_1, (512, 512), (1, 512), 0), out=buf523)
        del arg290_1
        buf527 = reinterpret_tensor(buf522, (8, 196, 512), (100352, 512, 1), 0); del buf522  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_41.run(buf507, buf523, arg291_1, arg292_1, arg293_1, buf527, 1568, 512, grid=grid(1568), stream=stream0)
        del arg292_1
        del arg293_1
        buf528 = reinterpret_tensor(buf505, (1568, 2048), (2048, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (1568, 512), (512, 1), 0), reinterpret_tensor(arg294_1, (512, 2048), (1, 512), 0), out=buf528)
        del arg294_1
        buf529 = reinterpret_tensor(buf528, (8, 196, 2048), (401408, 2048, 1), 0); del buf528  # reuse
        # Source Nodes: [x_405], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_37.run(buf529, arg295_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg295_1
        buf530 = reinterpret_tensor(buf527, (1568, 512), (512, 1), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg296_1, (2048, 512), (1, 2048), 0), out=buf530)
        del arg296_1
        del buf529
        buf531 = reinterpret_tensor(buf530, (8, 7, 7, 2, 2, 512), (100352, 14336, 1024, 512, 7168, 1), 0); del buf530  # reuse
        # Source Nodes: [x_413], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf531, buf507, buf523, arg291_1, arg297_1, 802816, grid=grid(802816), stream=stream0)
        del arg291_1
        del arg297_1
        del buf507
        buf535 = reinterpret_tensor(buf523, (8, 7, 7, 2048), (100352, 14336, 2048, 1), 0); del buf523  # reuse
        # Source Nodes: [x_414], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_44.run(buf531, arg298_1, arg299_1, buf535, 392, 2048, grid=grid(392), stream=stream0)
        del arg298_1
        del arg299_1
        del buf531
        buf536 = empty((392, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_416], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg300_1, (2048, 1024), (1, 2048), 0), out=buf536)
        del arg300_1
        del buf535
        buf540 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x_88], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_45.run(buf536, arg301_1, arg302_1, buf540, 392, 1024, grid=grid(392), stream=stream0)
        del arg301_1
        del arg302_1
        buf541 = empty((392, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg303_1, (1024, 3072), (1, 1024), 0), out=buf541)
        del arg303_1
        buf542 = reinterpret_tensor(buf540, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf540  # reuse
        # Source Nodes: [attn_110, q_45], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_46.run(buf541, arg304_1, buf542, 401408, grid=grid(401408), stream=stream0)
        buf543 = empty((8, 32, 32, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_110], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf541, arg304_1, buf543, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf544 = empty((256, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_110], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf543, (256, 32, 49), (1568, 49, 1), 0), out=buf544)
        buf547 = empty((8, 32, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_111, attn_112], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_48.run(buf544, arg362_1, arg22_1, buf547, 12544, 49, grid=grid(12544), stream=stream0)
        del arg22_1
        del arg362_1
        buf548 = reinterpret_tensor(buf543, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf543  # reuse
        # Source Nodes: [x_418], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf541, arg304_1, buf548, 401408, grid=grid(401408), stream=stream0)
        del arg304_1
        buf549 = reinterpret_tensor(buf542, (256, 49, 32), (1568, 32, 1), 0); del buf542  # reuse
        # Source Nodes: [x_418], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf547, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf548, (256, 49, 32), (1568, 32, 1), 0), out=buf549)
        buf550 = reinterpret_tensor(buf548, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf548  # reuse
        # Source Nodes: [x_419], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf549, buf550, 401408, grid=grid(401408), stream=stream0)
        buf551 = reinterpret_tensor(buf549, (392, 1024), (1024, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 1024), (1, 1024), 0), out=buf551)
        del arg305_1
        buf555 = reinterpret_tensor(buf550, (8, 49, 1024), (50176, 1024, 1), 0); del buf550  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_51.run(buf536, buf551, arg306_1, arg307_1, arg308_1, buf555, 392, 1024, grid=grid(392), stream=stream0)
        del arg307_1
        del arg308_1
        buf556 = reinterpret_tensor(buf107, (392, 4096), (4096, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 4096), (1, 1024), 0), out=buf556)
        del arg309_1
        buf557 = reinterpret_tensor(buf556, (8, 49, 4096), (200704, 4096, 1), 0); del buf556  # reuse
        # Source Nodes: [x_428], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf557, arg310_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg310_1
        buf558 = reinterpret_tensor(buf555, (392, 1024), (1024, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf557, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg311_1, (4096, 1024), (1, 4096), 0), out=buf558)
        del arg311_1
        buf559 = reinterpret_tensor(buf558, (8, 49, 1024), (50176, 1024, 1), 0); del buf558  # reuse
        buf563 = empty((8, 7, 7, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [shifted_x_92, x_433], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_53.run(buf559, buf536, buf551, arg306_1, arg312_1, arg313_1, arg314_1, buf563, 392, 1024, grid=grid(392), stream=stream0)
        del arg306_1
        del arg312_1
        del arg313_1
        del arg314_1
        del buf536
        buf564 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf563, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 3072), (1, 1024), 0), out=buf564)
        del arg315_1
        buf565 = reinterpret_tensor(buf563, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf563  # reuse
        # Source Nodes: [attn_114, q_47], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_46.run(buf564, arg316_1, buf565, 401408, grid=grid(401408), stream=stream0)
        buf566 = reinterpret_tensor(buf551, (8, 32, 32, 49), (50176, 1568, 49, 1), 0); del buf551  # reuse
        # Source Nodes: [attn_114], Original ATen: [aten.clone]
        triton_poi_fused_clone_47.run(buf564, arg316_1, buf566, 8192, 49, grid=grid(8192, 49), stream=stream0)
        buf567 = reinterpret_tensor(buf547, (256, 49, 49), (2401, 49, 1), 0); del buf547  # reuse
        # Source Nodes: [attn_114], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf565, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf566, (256, 32, 49), (1568, 49, 1), 0), out=buf567)
        buf570 = reinterpret_tensor(buf544, (8, 32, 49, 49), (76832, 2401, 49, 1), 0); del buf544  # reuse
        # Source Nodes: [attn_115, attn_116], Original ATen: [aten._softmax, aten.add]
        triton_per_fused__softmax_add_48.run(buf567, arg363_1, arg23_1, buf570, 12544, 49, grid=grid(12544), stream=stream0)
        del arg23_1
        del arg363_1
        del buf567
        buf571 = reinterpret_tensor(buf566, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf566  # reuse
        # Source Nodes: [x_436], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf564, arg316_1, buf571, 401408, grid=grid(401408), stream=stream0)
        del arg316_1
        del buf564
        buf572 = reinterpret_tensor(buf565, (256, 49, 32), (1568, 32, 1), 0); del buf565  # reuse
        # Source Nodes: [x_436], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf570, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf571, (256, 49, 32), (1568, 32, 1), 0), out=buf572)
        del buf570
        buf573 = reinterpret_tensor(buf571, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf571  # reuse
        # Source Nodes: [x_437], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf572, buf573, 401408, grid=grid(401408), stream=stream0)
        buf574 = reinterpret_tensor(buf572, (392, 1024), (1024, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), out=buf574)
        del arg317_1
        buf578 = reinterpret_tensor(buf573, (8, 49, 1024), (50176, 1024, 1), 0); del buf573  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___1___norm2], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_51.run(buf559, buf574, arg318_1, arg319_1, arg320_1, buf578, 392, 1024, grid=grid(392), stream=stream0)
        del arg319_1
        del arg320_1
        buf579 = reinterpret_tensor(buf557, (392, 4096), (4096, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 4096), (1, 1024), 0), out=buf579)
        del arg321_1
        buf580 = reinterpret_tensor(buf579, (8, 49, 4096), (200704, 4096, 1), 0); del buf579  # reuse
        # Source Nodes: [x_446], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_52.run(buf580, arg322_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg322_1
        buf581 = reinterpret_tensor(buf578, (392, 1024), (1024, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf580, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg323_1, (4096, 1024), (1, 4096), 0), out=buf581)
        del arg323_1
        del buf580
        buf582 = reinterpret_tensor(buf581, (8, 49, 1024), (50176, 1024, 1), 0); del buf581  # reuse
        buf583 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        buf584 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_451, x_456], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_54.run(buf582, buf559, buf574, arg318_1, arg324_1, buf583, buf584, 392, 1024, grid=grid(392), stream=stream0)
        del arg318_1
        del arg324_1
        del buf559
        del buf574
        buf586 = empty((8, 1024), device='cuda', dtype=torch.float32)
        buf587 = buf586; del buf586  # reuse
        # Source Nodes: [x_456, x_457], Original ATen: [aten.mean, aten.native_layer_norm]
        triton_per_fused_mean_native_layer_norm_55.run(buf587, buf582, buf583, buf584, arg325_1, arg326_1, 8192, 49, grid=grid(8192), stream=stream0)
        del arg325_1
        del arg326_1
        del buf582
        del buf583
        del buf584
        buf588 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_456, x_457, x_461], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
        extern_kernels.addmm(arg328_1, buf587, reinterpret_tensor(arg327_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf588)
        del arg327_1
        del arg328_1
        return (buf588, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((169, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((169, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((169, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((169, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg330_1 = rand_strided((64, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg332_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg333_1 = rand_strided((16, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg335_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg336_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg338_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg339_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg341_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg342_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg344_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg345_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg347_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg348_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg350_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg351_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg353_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg354_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg356_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg357_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg359_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg360_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg362_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg363_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg364_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swin_base_patch4_window7_224', benchmark_compiled_module)
