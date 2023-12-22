
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


# kernel path: /tmp/torchinductor_youkaichao/go/cgobra7jxvlfxcm2mhj6mppwxnbfso4ii2hwktpzddntcwq3co4l.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbdwyfzzbpxnmshpgrbgeo5psffwclhqkdxe67snw6ildtogvld.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (13*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjjn22nlxhais5anmatzda6cxn5ypwglgiysgfpsbvzq4io3wgd.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# shortcut => relu
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosywsi5oqitlt5ec3auifqyrhshibi2nc4nokudzzwf57so3ouv.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 312
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (301056*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxve3xugcbh2gao5medgsbm3is5226q5h6mgo7ofoedtm42znnr.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (13*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
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


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4hi6xdigthisd2j2jzivbh5vbno43awcf3lw2ahlivvotitxzm.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_11 => relu_1
# x_7 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lqnnlu4mjn4kq4bhwstmucablclofcmyosay3yvfkkurbwzus7.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpuhicpas2ewzh2pwp3sdnoy4uam2xhtade2nvqbkr2y4oey27ff.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (24*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crhl4npmxcowub3mleqcg7kibbjou5zuvxhliwxose3zkkfumd2b.py
# Source Nodes: [x_13, x_17, x_se], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
# x_13 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_17 => relu_2
# x_se => mean
triton_red_fused__native_batch_norm_legit_functional_mean_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_mean_relu_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 24
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 25088.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = tl.math.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = triton_helpers.maximum(0, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tl.store(out_ptr0 + (r2 + (3136*x3)), tmp14, rmask & xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = 3136.0
    tmp19 = tmp16 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zdlt2d3hlzij4qmmfyvhu3ezkinmv56nemh2vlukhbpdhaf2v7.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_3
# x_se_2 => relu_3
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4w5jwuzew4cwsp3v7n4x76sem56vjokpymqk5gwizwbtextz4hr.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_4
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7trzmc2focl25bgttetmykf7wwdfoqzagbuqyways7bkg4ag6go.py
# Source Nodes: [sigmoid, x_18], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid => sigmoid
# x_18 => mul_21
triton_poi_fused_mul_sigmoid_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb23e7fiwk5nhqdlejs66hpzqzewrend46wovmigpayenp6r5ubb.py
# Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_1 => relu_4
# x_20 => add_16, add_19, mul_22, mul_28, rsqrt_3, sub_3, var_mean_3
# x_26 => add_21, add_24, mul_29, mul_35, rsqrt_4, sub_4, var_mean_4
# x_30 => add_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7geebtp6kcwcj2e22ejjjyjjl4fqktt6b75qlywptcrvhhwr6ph.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 224
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 56
    x1 = (xindex // 56)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (175616*(r2 // 3136)) + (351232*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdrsbyoxfjawsfzhnj77ue5zpjvjmj5iihaguiacvrqcj6yrrmv.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => add_27, add_28, add_29, mul_37, mul_38, mul_39, mul_40, mul_41, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (56*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (56*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
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


# kernel path: /tmp/torchinductor_youkaichao/os/cosg5yqco5322godcblnaqymee3azzmfxre322vofsxgh2awb64x.py
# Source Nodes: [x_35, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_35 => add_27, add_30, mul_36, mul_42, rsqrt_5, sub_5, var_mean_5
# x_39 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 56
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbq4krg5dhz6gktlcib4baatvpiegbmdes5f7g3xw4dwt2ddlbw.py
# Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
# x_41 => add_32, add_33, add_34, mul_44, mul_45, mul_46, mul_47, mul_48, rsqrt_6, squeeze_19, var_mean_6
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 6272.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001594642002871
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4f3qcackii4za3kt6tgyvwgbuqax5ogshvr35qn5o7mc44cany.py
# Source Nodes: [x_41, x_45, x_se_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
# x_41 => add_32, add_35, mul_43, mul_49, rsqrt_6, sub_6, var_mean_6
# x_45 => relu_6
# x_se_4 => mean_1
triton_per_fused__native_batch_norm_legit_functional_mean_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel):
    xnumel = 448
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 56
    tmp0 = tl.load(in_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lj/clj5sdte7ouvvwgceatybkvgsb2olmzula2dzxmju3acmfvxgy6n.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_9
# x_se_6 => relu_7
triton_poi_fused_convolution_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2ktysqxpm6ghmlovjzgbr65el3iun7rhmq2udaqalh62uzyk6l.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_10
triton_poi_fused_convolution_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 56
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3t2wnszczttcgeh23gkmz3w2dfl66vh565zdjdpwqmpkqf6mhu.py
# Source Nodes: [sigmoid_1, x_46], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_1 => sigmoid_1
# x_46 => mul_50
triton_poi_fused_mul_sigmoid_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcdcfokctrgo662qts3zwn6pnkusbdrbm3hzxl7tkujppc7jom2.py
# Source Nodes: [shortcut_2, x_48, x_54, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_2 => relu_8
# x_48 => add_37, add_40, mul_51, mul_57, rsqrt_7, sub_7, var_mean_7
# x_54 => add_42, add_45, mul_58, mul_64, rsqrt_8, sub_8, var_mean_8
# x_58 => add_46
triton_poi_fused__native_batch_norm_legit_functional_add_relu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwlsu6jwkwpx67pgknmvsqlqnio5towwn6ugltq3lpzczdswx3t.py
# Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
# x_63 => add_48, add_49, add_50, mul_66, mul_67, mul_68, mul_69, mul_70, rsqrt_9, squeeze_28, var_mean_9
triton_red_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (119168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 6272.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001594642002871
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ud/cudldyjpyx6f3nmm35qudjkrccc3w3je6ayz6wbcalekdokmzjfk.py
# Source Nodes: [x_63, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_63 => add_48, add_51, mul_65, mul_71, rsqrt_9, sub_9, var_mean_9
# x_67 => relu_9
triton_poi_fused__native_batch_norm_legit_functional_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckv5742dqvr74mszpwx7oj4ex5ufgmf2cigyz2gmdu3ce4u4bp4.py
# Source Nodes: [x_69], Original ATen: [aten._native_batch_norm_legit_functional]
# x_69 => add_53, add_54, add_55, mul_73, mul_74, mul_75, mul_76, mul_77, rsqrt_10, squeeze_31, var_mean_10
triton_red_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (29792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1568.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0006381620931717
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlt6xhlnuouqfjuffjb42nfeggl6v5oemnufnzuq5f4ugqmdq7s.py
# Source Nodes: [x_69, x_73, x_se_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
# x_69 => add_53, add_56, mul_72, mul_78, rsqrt_10, sub_10, var_mean_10
# x_73 => relu_10
# x_se_8 => mean_2
triton_per_fused__native_batch_norm_legit_functional_mean_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cnclsnn6yh3jc3zw2v2gn3qhfnmw3stkfcpsdjpofiux7ko6xoyq.py
# Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
# x_se_10 => relu_11
# x_se_9 => convolution_15
triton_poi_fused_convolution_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 14
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrc7r3pmbhdinhvvsolu6s6i3r6jhonpj5dmgsckw4sqns6tsnd.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_16
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkxyy457kyltejg4qonk5klxjkuxyr36jdr37hzdiu57qtvruow.py
# Source Nodes: [sigmoid_2, x_74], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_2 => sigmoid_2
# x_74 => mul_79
triton_poi_fused_mul_sigmoid_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtlcnf5msdt7xltf3dzhosd7kbazxiqman4waap2ihiqg2yfxi6.py
# Source Nodes: [shortcut_3, x_76, x_82, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_3 => relu_12
# x_76 => add_58, add_61, mul_80, mul_86, rsqrt_11, sub_11, var_mean_11
# x_82 => add_63, add_66, mul_87, mul_93, rsqrt_12, sub_12, var_mean_12
# x_86 => add_67
triton_poi_fused__native_batch_norm_legit_functional_add_relu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdytvz2why7boju4mff5yhltyatdrvssniefhqe7vnbhtktkk5v.py
# Source Nodes: [x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_90 => add_69, add_72, mul_100, mul_94, rsqrt_13, sub_13, var_mean_13
# x_94 => relu_13
triton_poi_fused__native_batch_norm_legit_functional_relu_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjj7dk75soxtuspeddyfuge2t4fxtwtlemegxhjls22sia3a2ud.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_21
# x_se_14 => relu_15
triton_poi_fused_convolution_relu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 38
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6se6k7e3dvtee2jplfgqayzghtlixqtiaer4pzckybllotls5ac.py
# Source Nodes: [shortcut_4, x_103, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_4 => relu_16
# x_103 => add_79, add_82, mul_109, mul_115, rsqrt_15, sub_15, var_mean_15
# x_108 => add_83
triton_poi_fused__native_batch_norm_legit_functional_add_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/clezdff3j2owd25upq73va2gclnr77td4fkurzjadjysspetnuua.py
# Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
# x_157 => add_117, add_118, add_119, mul_161, mul_162, mul_163, mul_164, mul_165, rsqrt_22, squeeze_67, var_mean_22
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 368
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (72128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1568.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0006381620931717
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calro5mjqc2voezq2upeoztwjcwcrjochjwejhe6wnq4lnny6qgg.py
# Source Nodes: [x_157, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_157 => add_117, add_120, mul_160, mul_166, rsqrt_22, sub_22, var_mean_22
# x_161 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clozqzkgqpswsfedwcxj5h5u6s5avx7ougfvld2qfgdv26js3fpg.py
# Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
# x_163 => add_122, add_123, add_124, mul_168, mul_169, mul_170, mul_171, mul_172, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 368
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (18032*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0025575447570332
    tmp29 = tmp18 * tmp28
    tmp30 = tmp29 * tmp22
    tmp32 = tmp31 * tmp25
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvsnszmikfkhqoza5drasrzjwt7jlu653g424b5xl27j7wrhuew.py
# Source Nodes: [x_163, x_167, x_se_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
# x_163 => add_122, add_125, mul_167, mul_173, rsqrt_23, sub_23, var_mean_23
# x_167 => relu_26
# x_se_24 => mean_6
triton_per_fused__native_batch_norm_legit_functional_mean_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tl.store(out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yh3t5gaci2634r3ahnfbjb6oud23ty24py6wbeqal2wkyursor.py
# Source Nodes: [x_se_27], Original ATen: [aten.convolution]
# x_se_27 => convolution_37
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3f5vcfn6dcf5h44ud32tgzxwtsb4fob5b6iindtk3fmylpt3jay.py
# Source Nodes: [sigmoid_6, x_168], Original ATen: [aten.mul, aten.sigmoid]
# sigmoid_6 => sigmoid_6
# x_168 => mul_174
triton_poi_fused_mul_sigmoid_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdnkyjoccek3gaciiuuzaw53w6akyptrsvdi6ar23tzwdz66wrc.py
# Source Nodes: [shortcut_7, x_170, x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_7 => relu_28
# x_170 => add_127, add_130, mul_175, mul_181, rsqrt_24, sub_24, var_mean_24
# x_176 => add_132, add_135, mul_182, mul_188, rsqrt_25, sub_25, var_mean_25
# x_180 => add_136
triton_poi_fused__native_batch_norm_legit_functional_add_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chp4hknonleq6n7qdpb4z4hpqxvi4odbqlytrw777pklomqkfogq.py
# Source Nodes: [x_184, x_188], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_184 => add_138, add_141, mul_189, mul_195, rsqrt_26, sub_26, var_mean_26
# x_188 => relu_29
triton_poi_fused__native_batch_norm_legit_functional_relu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dfkqxt6iubqq2d5x7nandrbwlsmb2oaapv3gtmtauyrg64dqog.py
# Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
# x_se_29 => convolution_42
# x_se_30 => relu_31
triton_poi_fused_convolution_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 92
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpkafnyxff467zujri6ibdivfazj5gy4psrqiakewsji3mii7f2.py
# Source Nodes: [shortcut_8, x_197, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_8 => relu_32
# x_197 => add_148, add_151, mul_204, mul_210, rsqrt_28, sub_28, var_mean_28
# x_202 => add_152
triton_poi_fused__native_batch_norm_legit_functional_add_relu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 368
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqktlk7dixzy2pvpdj374vvny3lk3trt2nlf6w7a3eukhmxszvcd.py
# Source Nodes: [x_307, x_312, x_315, x_318, x_320], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# x_307 => add_228, add_231, mul_314, mul_320, rsqrt_43, sub_43, var_mean_43
# x_312 => add_232
# x_315 => relu_52
# x_318 => mean_13
# x_320 => view
triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 368
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 49.0
    tmp24 = tmp22 / tmp23
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmjhwaw7nzxurn6hpsmm3mscnqpoyd6764qtiulrxowrjvnbd5z.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (24, ), (1, ))
    assert_size_stride(primals_8, (24, ), (1, ))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_10, (24, ), (1, ))
    assert_size_stride(primals_11, (56, ), (1, ))
    assert_size_stride(primals_12, (56, ), (1, ))
    assert_size_stride(primals_13, (56, ), (1, ))
    assert_size_stride(primals_14, (56, ), (1, ))
    assert_size_stride(primals_15, (56, ), (1, ))
    assert_size_stride(primals_16, (56, ), (1, ))
    assert_size_stride(primals_17, (56, ), (1, ))
    assert_size_stride(primals_18, (56, ), (1, ))
    assert_size_stride(primals_19, (152, ), (1, ))
    assert_size_stride(primals_20, (152, ), (1, ))
    assert_size_stride(primals_21, (152, ), (1, ))
    assert_size_stride(primals_22, (152, ), (1, ))
    assert_size_stride(primals_23, (152, ), (1, ))
    assert_size_stride(primals_24, (152, ), (1, ))
    assert_size_stride(primals_25, (152, ), (1, ))
    assert_size_stride(primals_26, (152, ), (1, ))
    assert_size_stride(primals_27, (152, ), (1, ))
    assert_size_stride(primals_28, (152, ), (1, ))
    assert_size_stride(primals_29, (152, ), (1, ))
    assert_size_stride(primals_30, (152, ), (1, ))
    assert_size_stride(primals_31, (152, ), (1, ))
    assert_size_stride(primals_32, (152, ), (1, ))
    assert_size_stride(primals_33, (152, ), (1, ))
    assert_size_stride(primals_34, (152, ), (1, ))
    assert_size_stride(primals_35, (152, ), (1, ))
    assert_size_stride(primals_36, (152, ), (1, ))
    assert_size_stride(primals_37, (152, ), (1, ))
    assert_size_stride(primals_38, (152, ), (1, ))
    assert_size_stride(primals_39, (152, ), (1, ))
    assert_size_stride(primals_40, (152, ), (1, ))
    assert_size_stride(primals_41, (152, ), (1, ))
    assert_size_stride(primals_42, (152, ), (1, ))
    assert_size_stride(primals_43, (152, ), (1, ))
    assert_size_stride(primals_44, (152, ), (1, ))
    assert_size_stride(primals_45, (368, ), (1, ))
    assert_size_stride(primals_46, (368, ), (1, ))
    assert_size_stride(primals_47, (368, ), (1, ))
    assert_size_stride(primals_48, (368, ), (1, ))
    assert_size_stride(primals_49, (368, ), (1, ))
    assert_size_stride(primals_50, (368, ), (1, ))
    assert_size_stride(primals_51, (368, ), (1, ))
    assert_size_stride(primals_52, (368, ), (1, ))
    assert_size_stride(primals_53, (368, ), (1, ))
    assert_size_stride(primals_54, (368, ), (1, ))
    assert_size_stride(primals_55, (368, ), (1, ))
    assert_size_stride(primals_56, (368, ), (1, ))
    assert_size_stride(primals_57, (368, ), (1, ))
    assert_size_stride(primals_58, (368, ), (1, ))
    assert_size_stride(primals_59, (368, ), (1, ))
    assert_size_stride(primals_60, (368, ), (1, ))
    assert_size_stride(primals_61, (368, ), (1, ))
    assert_size_stride(primals_62, (368, ), (1, ))
    assert_size_stride(primals_63, (368, ), (1, ))
    assert_size_stride(primals_64, (368, ), (1, ))
    assert_size_stride(primals_65, (368, ), (1, ))
    assert_size_stride(primals_66, (368, ), (1, ))
    assert_size_stride(primals_67, (368, ), (1, ))
    assert_size_stride(primals_68, (368, ), (1, ))
    assert_size_stride(primals_69, (368, ), (1, ))
    assert_size_stride(primals_70, (368, ), (1, ))
    assert_size_stride(primals_71, (368, ), (1, ))
    assert_size_stride(primals_72, (368, ), (1, ))
    assert_size_stride(primals_73, (368, ), (1, ))
    assert_size_stride(primals_74, (368, ), (1, ))
    assert_size_stride(primals_75, (368, ), (1, ))
    assert_size_stride(primals_76, (368, ), (1, ))
    assert_size_stride(primals_77, (368, ), (1, ))
    assert_size_stride(primals_78, (368, ), (1, ))
    assert_size_stride(primals_79, (368, ), (1, ))
    assert_size_stride(primals_80, (368, ), (1, ))
    assert_size_stride(primals_81, (368, ), (1, ))
    assert_size_stride(primals_82, (368, ), (1, ))
    assert_size_stride(primals_83, (368, ), (1, ))
    assert_size_stride(primals_84, (368, ), (1, ))
    assert_size_stride(primals_85, (368, ), (1, ))
    assert_size_stride(primals_86, (368, ), (1, ))
    assert_size_stride(primals_87, (368, ), (1, ))
    assert_size_stride(primals_88, (368, ), (1, ))
    assert_size_stride(primals_89, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_90, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_91, (24, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_92, (8, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_93, (8, ), (1, ))
    assert_size_stride(primals_94, (24, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_95, (24, ), (1, ))
    assert_size_stride(primals_96, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_97, (24, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_98, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_99, (56, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_100, (6, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_101, (6, ), (1, ))
    assert_size_stride(primals_102, (56, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_103, (56, ), (1, ))
    assert_size_stride(primals_104, (56, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_105, (56, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_106, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_107, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_108, (14, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_109, (14, ), (1, ))
    assert_size_stride(primals_110, (152, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_111, (152, ), (1, ))
    assert_size_stride(primals_112, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_113, (152, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_114, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_115, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_116, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_117, (38, ), (1, ))
    assert_size_stride(primals_118, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_119, (152, ), (1, ))
    assert_size_stride(primals_120, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_121, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_122, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_123, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_124, (38, ), (1, ))
    assert_size_stride(primals_125, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_126, (152, ), (1, ))
    assert_size_stride(primals_127, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_128, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_129, (152, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_130, (38, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_131, (38, ), (1, ))
    assert_size_stride(primals_132, (152, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_133, (152, ), (1, ))
    assert_size_stride(primals_134, (152, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_135, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_136, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_137, (38, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_138, (38, ), (1, ))
    assert_size_stride(primals_139, (368, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_140, (368, ), (1, ))
    assert_size_stride(primals_141, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_142, (368, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_143, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_144, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_145, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_146, (92, ), (1, ))
    assert_size_stride(primals_147, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_148, (368, ), (1, ))
    assert_size_stride(primals_149, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_150, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_151, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_152, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_153, (92, ), (1, ))
    assert_size_stride(primals_154, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_155, (368, ), (1, ))
    assert_size_stride(primals_156, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_157, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_158, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_159, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_160, (92, ), (1, ))
    assert_size_stride(primals_161, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_162, (368, ), (1, ))
    assert_size_stride(primals_163, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_164, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_165, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_166, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_167, (92, ), (1, ))
    assert_size_stride(primals_168, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_169, (368, ), (1, ))
    assert_size_stride(primals_170, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_171, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_172, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_173, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_174, (92, ), (1, ))
    assert_size_stride(primals_175, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_176, (368, ), (1, ))
    assert_size_stride(primals_177, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_178, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_179, (368, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_180, (92, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_181, (92, ), (1, ))
    assert_size_stride(primals_182, (368, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(primals_183, (368, ), (1, ))
    assert_size_stride(primals_184, (368, 368, 1, 1), (368, 1, 1, 1))
    assert_size_stride(primals_185, (1000, 368), (368, 1))
    assert_size_stride(primals_186, (1000, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (32, ), (1, ))
    assert_size_stride(primals_189, (32, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (24, ), (1, ))
    assert_size_stride(primals_195, (24, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (24, ), (1, ))
    assert_size_stride(primals_198, (24, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_201, (24, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (56, ), (1, ))
    assert_size_stride(primals_204, (56, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (56, ), (1, ))
    assert_size_stride(primals_207, (56, ), (1, ))
    assert_size_stride(primals_208, (), ())
    assert_size_stride(primals_209, (56, ), (1, ))
    assert_size_stride(primals_210, (56, ), (1, ))
    assert_size_stride(primals_211, (), ())
    assert_size_stride(primals_212, (56, ), (1, ))
    assert_size_stride(primals_213, (56, ), (1, ))
    assert_size_stride(primals_214, (), ())
    assert_size_stride(primals_215, (152, ), (1, ))
    assert_size_stride(primals_216, (152, ), (1, ))
    assert_size_stride(primals_217, (), ())
    assert_size_stride(primals_218, (152, ), (1, ))
    assert_size_stride(primals_219, (152, ), (1, ))
    assert_size_stride(primals_220, (), ())
    assert_size_stride(primals_221, (152, ), (1, ))
    assert_size_stride(primals_222, (152, ), (1, ))
    assert_size_stride(primals_223, (), ())
    assert_size_stride(primals_224, (152, ), (1, ))
    assert_size_stride(primals_225, (152, ), (1, ))
    assert_size_stride(primals_226, (), ())
    assert_size_stride(primals_227, (152, ), (1, ))
    assert_size_stride(primals_228, (152, ), (1, ))
    assert_size_stride(primals_229, (), ())
    assert_size_stride(primals_230, (152, ), (1, ))
    assert_size_stride(primals_231, (152, ), (1, ))
    assert_size_stride(primals_232, (), ())
    assert_size_stride(primals_233, (152, ), (1, ))
    assert_size_stride(primals_234, (152, ), (1, ))
    assert_size_stride(primals_235, (), ())
    assert_size_stride(primals_236, (152, ), (1, ))
    assert_size_stride(primals_237, (152, ), (1, ))
    assert_size_stride(primals_238, (), ())
    assert_size_stride(primals_239, (152, ), (1, ))
    assert_size_stride(primals_240, (152, ), (1, ))
    assert_size_stride(primals_241, (), ())
    assert_size_stride(primals_242, (152, ), (1, ))
    assert_size_stride(primals_243, (152, ), (1, ))
    assert_size_stride(primals_244, (), ())
    assert_size_stride(primals_245, (152, ), (1, ))
    assert_size_stride(primals_246, (152, ), (1, ))
    assert_size_stride(primals_247, (), ())
    assert_size_stride(primals_248, (152, ), (1, ))
    assert_size_stride(primals_249, (152, ), (1, ))
    assert_size_stride(primals_250, (), ())
    assert_size_stride(primals_251, (152, ), (1, ))
    assert_size_stride(primals_252, (152, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (368, ), (1, ))
    assert_size_stride(primals_255, (368, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (368, ), (1, ))
    assert_size_stride(primals_258, (368, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (368, ), (1, ))
    assert_size_stride(primals_261, (368, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (368, ), (1, ))
    assert_size_stride(primals_264, (368, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (368, ), (1, ))
    assert_size_stride(primals_267, (368, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (368, ), (1, ))
    assert_size_stride(primals_270, (368, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (368, ), (1, ))
    assert_size_stride(primals_273, (368, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (368, ), (1, ))
    assert_size_stride(primals_276, (368, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (368, ), (1, ))
    assert_size_stride(primals_279, (368, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (368, ), (1, ))
    assert_size_stride(primals_282, (368, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (368, ), (1, ))
    assert_size_stride(primals_285, (368, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (368, ), (1, ))
    assert_size_stride(primals_288, (368, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (368, ), (1, ))
    assert_size_stride(primals_291, (368, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (368, ), (1, ))
    assert_size_stride(primals_294, (368, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (368, ), (1, ))
    assert_size_stride(primals_297, (368, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (368, ), (1, ))
    assert_size_stride(primals_300, (368, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (368, ), (1, ))
    assert_size_stride(primals_303, (368, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (368, ), (1, ))
    assert_size_stride(primals_306, (368, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (368, ), (1, ))
    assert_size_stride(primals_309, (368, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (368, ), (1, ))
    assert_size_stride(primals_312, (368, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (368, ), (1, ))
    assert_size_stride(primals_315, (368, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (368, ), (1, ))
    assert_size_stride(primals_318, (368, ), (1, ))
    assert_size_stride(primals_319, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_319, primals_89, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf1 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 416, 7720, grid=grid(416), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf7 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_188, primals_189, buf4, buf5, buf7, primals_188, primals_189, 32, 13, grid=grid(32), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_188
        del primals_189
        buf8 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_1, primals_2, buf8, 3211264, grid=grid(3211264), stream=stream0)
        del buf5
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 24, 112, 112), (301056, 12544, 112, 1))
        buf10 = empty_strided((1, 24, 1, 1, 13), (312, 13, 312, 312, 1), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 24, 1, 1, 13), (312, 13, 312, 312, 1), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 24, 1, 1, 13), (312, 13, 312, 312, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf9, buf10, buf11, buf12, 312, 7720, grid=grid(312), stream=stream0)
        buf13 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf16 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf10, buf11, buf12, primals_191, primals_192, buf13, buf14, buf16, primals_191, primals_192, 24, 13, grid=grid(24), stream=stream0)
        del buf10
        del buf11
        del buf12
        del primals_191
        del primals_192
        buf17 = empty((8, 24, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf9, buf13, buf14, primals_3, primals_4, buf17, 2408448, grid=grid(2408448), stream=stream0)
        del primals_4
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_91, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf19 = empty_strided((1, 24, 1, 1, 4), (96, 1, 96, 96, 24), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 24, 1, 1, 4), (96, 1, 96, 96, 24), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 24, 1, 1, 4), (96, 1, 96, 96, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf18, buf19, buf20, buf21, 96, 6272, grid=grid(96), stream=stream0)
        buf22 = buf14; del buf14  # reuse
        buf23 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf25 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_7.run(buf19, buf20, buf21, primals_194, primals_195, buf22, buf23, buf25, primals_194, primals_195, 24, 4, grid=grid(24), stream=stream0)
        del primals_194
        del primals_195
        buf26 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        buf27 = empty_strided((8, 24, 1, 1), (24, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf28 = reinterpret_tensor(buf27, (8, 24, 1, 1), (24, 1, 1, 1), 0); del buf27  # reuse
        # Source Nodes: [x_13, x_17, x_se], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_red_fused__native_batch_norm_legit_functional_mean_relu_8.run(buf28, buf18, buf22, buf23, primals_5, primals_6, buf26, 192, 3136, grid=grid(192), stream=stream0)
        del primals_6
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 8, 1, 1), (8, 1, 1, 1))
        buf30 = buf29; del buf29  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(buf30, primals_93, 64, grid=grid(64), stream=stream0)
        del primals_93
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 24, 1, 1), (24, 1, 1, 1))
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf32, primals_95, 192, grid=grid(192), stream=stream0)
        del primals_95
        buf33 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_18], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_11.run(buf26, buf32, buf33, 602112, grid=grid(602112), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf35 = buf21; del buf21  # reuse
        buf36 = buf20; del buf20  # reuse
        buf37 = buf19; del buf19  # reuse
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf34, buf35, buf36, buf37, 96, 6272, grid=grid(96), stream=stream0)
        buf38 = buf23; del buf23  # reuse
        buf39 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf41 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_7.run(buf35, buf36, buf37, primals_197, primals_198, buf38, buf39, buf41, primals_197, primals_198, 24, 4, grid=grid(24), stream=stream0)
        del primals_197
        del primals_198
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf8, primals_97, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf43 = buf37; del buf37  # reuse
        buf44 = buf36; del buf36  # reuse
        buf45 = buf35; del buf35  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf42, buf43, buf44, buf45, 96, 6272, grid=grid(96), stream=stream0)
        buf46 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf47 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cuda', dtype=torch.float32)
        buf49 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_7.run(buf43, buf44, buf45, primals_200, primals_201, buf46, buf47, buf49, primals_200, primals_201, 24, 4, grid=grid(24), stream=stream0)
        del buf43
        del buf44
        del buf45
        del primals_200
        del primals_201
        buf50 = empty((8, 24, 56, 56), device='cuda', dtype=torch.float32)
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [shortcut_1, x_20, x_26, x_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_12.run(buf51, buf34, buf38, buf39, primals_7, primals_8, buf42, buf46, buf47, primals_9, primals_10, 602112, grid=grid(602112), stream=stream0)
        del buf39
        del buf47
        del primals_10
        del primals_8
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 56, 56, 56), (175616, 3136, 56, 1))
        buf53 = empty_strided((1, 56, 1, 1, 4), (224, 1, 224, 224, 56), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((1, 56, 1, 1, 4), (224, 1, 224, 224, 56), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((1, 56, 1, 1, 4), (224, 1, 224, 224, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf52, buf53, buf54, buf55, 224, 6272, grid=grid(224), stream=stream0)
        buf56 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf57 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf59 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_14.run(buf53, buf54, buf55, primals_203, primals_204, buf56, buf57, buf59, primals_203, primals_204, 56, 4, grid=grid(56), stream=stream0)
        del buf53
        del buf54
        del buf55
        del primals_203
        del primals_204
        buf60 = empty((8, 56, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_15.run(buf52, buf56, buf57, primals_11, primals_12, buf60, 1404928, grid=grid(1404928), stream=stream0)
        del primals_12
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_99, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=7, bias=None)
        assert_size_stride(buf61, (8, 56, 28, 28), (43904, 784, 28, 1))
        buf62 = buf57; del buf57  # reuse
        buf63 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf65 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf61, primals_206, primals_207, buf62, buf63, buf65, primals_206, primals_207, 56, 6272, grid=grid(56), stream=stream0)
        del primals_206
        del primals_207
        buf66 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((8, 56, 1, 1), (56, 1, 448, 448), device='cuda', dtype=torch.float32)
        buf68 = reinterpret_tensor(buf67, (8, 56, 1, 1), (56, 1, 1, 1), 0); del buf67  # reuse
        # Source Nodes: [x_41, x_45, x_se_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_17.run(buf68, buf61, buf62, buf63, primals_13, primals_14, buf66, 448, 784, grid=grid(448), stream=stream0)
        del primals_14
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 6, 1, 1), (6, 1, 1, 1))
        buf70 = buf69; del buf69  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_18.run(buf70, primals_101, 48, grid=grid(48), stream=stream0)
        del primals_101
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 56, 1, 1), (56, 1, 1, 1))
        buf72 = buf71; del buf71  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf72, primals_103, 448, grid=grid(448), stream=stream0)
        del primals_103
        buf73 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_46], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_20.run(buf66, buf72, buf73, 351232, grid=grid(351232), stream=stream0)
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 56, 28, 28), (43904, 784, 28, 1))
        buf75 = buf63; del buf63  # reuse
        buf76 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf78 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf74, primals_209, primals_210, buf75, buf76, buf78, primals_209, primals_210, 56, 6272, grid=grid(56), stream=stream0)
        del primals_209
        del primals_210
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf51, primals_105, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 56, 28, 28), (43904, 784, 28, 1))
        buf80 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf81 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cuda', dtype=torch.float32)
        buf83 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf79, primals_212, primals_213, buf80, buf81, buf83, primals_212, primals_213, 56, 6272, grid=grid(56), stream=stream0)
        del primals_212
        del primals_213
        buf84 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        buf85 = buf84; del buf84  # reuse
        # Source Nodes: [shortcut_2, x_48, x_54, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_21.run(buf85, buf74, buf75, buf76, primals_15, primals_16, buf79, buf80, buf81, primals_17, primals_18, 351232, grid=grid(351232), stream=stream0)
        del buf76
        del buf81
        del primals_16
        del primals_18
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 152, 28, 28), (119168, 784, 28, 1))
        buf87 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf88 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf90 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_22.run(buf86, primals_215, primals_216, buf87, buf88, buf90, primals_215, primals_216, 152, 6272, grid=grid(152), stream=stream0)
        del primals_215
        del primals_216
        buf91 = empty((8, 152, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf86, buf87, buf88, primals_19, primals_20, buf91, 953344, grid=grid(953344), stream=stream0)
        del primals_20
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_107, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf92, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf93 = buf88; del buf88  # reuse
        buf94 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf96 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf92, primals_218, primals_219, buf93, buf94, buf96, primals_218, primals_219, 152, 1568, grid=grid(152), stream=stream0)
        del primals_218
        del primals_219
        buf97 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf98 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf99 = reinterpret_tensor(buf98, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf98  # reuse
        # Source Nodes: [x_69, x_73, x_se_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_25.run(buf99, buf92, buf93, buf94, primals_21, primals_22, buf97, 1216, 196, grid=grid(1216), stream=stream0)
        del primals_22
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 14, 1, 1), (14, 1, 1, 1))
        buf101 = buf100; del buf100  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_26.run(buf101, primals_109, 112, grid=grid(112), stream=stream0)
        del primals_109
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 152, 1, 1), (152, 1, 1, 1))
        buf103 = buf102; del buf102  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf103, primals_111, 1216, grid=grid(1216), stream=stream0)
        del primals_111
        buf104 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_74], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_28.run(buf97, buf103, buf104, 238336, grid=grid(238336), stream=stream0)
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf106 = buf94; del buf94  # reuse
        buf107 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf109 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf105, primals_221, primals_222, buf106, buf107, buf109, primals_221, primals_222, 152, 1568, grid=grid(152), stream=stream0)
        del primals_221
        del primals_222
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf85, primals_113, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf111 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf112 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf114 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf110, primals_224, primals_225, buf111, buf112, buf114, primals_224, primals_225, 152, 1568, grid=grid(152), stream=stream0)
        del primals_224
        del primals_225
        buf115 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [shortcut_3, x_76, x_82, x_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf116, buf105, buf106, buf107, primals_23, primals_24, buf110, buf111, buf112, primals_25, primals_26, 238336, grid=grid(238336), stream=stream0)
        del primals_24
        del primals_26
        # Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf118 = buf112; del buf112  # reuse
        buf119 = buf107; del buf107  # reuse
        buf121 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf117, primals_227, primals_228, buf118, buf119, buf121, primals_227, primals_228, 152, 1568, grid=grid(152), stream=stream0)
        del primals_227
        del primals_228
        buf122 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_90, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf117, buf118, buf119, primals_27, primals_28, buf122, 238336, grid=grid(238336), stream=stream0)
        del primals_28
        # Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf123, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf124 = buf119; del buf119  # reuse
        buf125 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf127 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf123, primals_230, primals_231, buf124, buf125, buf127, primals_230, primals_231, 152, 1568, grid=grid(152), stream=stream0)
        del primals_230
        del primals_231
        buf128 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf129 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf130 = reinterpret_tensor(buf129, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf129  # reuse
        # Source Nodes: [x_100, x_96, x_se_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_25.run(buf130, buf123, buf124, buf125, primals_29, primals_30, buf128, 1216, 196, grid=grid(1216), stream=stream0)
        del primals_30
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 38, 1, 1), (38, 1, 1, 1))
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_31.run(buf132, primals_117, 304, grid=grid(304), stream=stream0)
        del primals_117
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 152, 1, 1), (152, 1, 1, 1))
        buf134 = buf133; del buf133  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf134, primals_119, 1216, grid=grid(1216), stream=stream0)
        del primals_119
        buf135 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_101], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_28.run(buf128, buf134, buf135, 238336, grid=grid(238336), stream=stream0)
        # Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf137 = buf125; del buf125  # reuse
        buf138 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf140 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf136, primals_233, primals_234, buf137, buf138, buf140, primals_233, primals_234, 152, 1568, grid=grid(152), stream=stream0)
        del primals_233
        del primals_234
        buf141 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_4, x_103, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_32.run(buf136, buf137, buf138, primals_31, primals_32, buf116, buf141, 238336, grid=grid(238336), stream=stream0)
        del primals_32
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf143 = buf138; del buf138  # reuse
        buf144 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf146 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf142, primals_236, primals_237, buf143, buf144, buf146, primals_236, primals_237, 152, 1568, grid=grid(152), stream=stream0)
        del primals_236
        del primals_237
        buf147 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112, x_116], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf142, buf143, buf144, primals_33, primals_34, buf147, 238336, grid=grid(238336), stream=stream0)
        del primals_34
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf148, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf149 = buf144; del buf144  # reuse
        buf150 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf152 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf148, primals_239, primals_240, buf149, buf150, buf152, primals_239, primals_240, 152, 1568, grid=grid(152), stream=stream0)
        del primals_239
        del primals_240
        buf153 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf154 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf155 = reinterpret_tensor(buf154, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf154  # reuse
        # Source Nodes: [x_118, x_122, x_se_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_25.run(buf155, buf148, buf149, buf150, primals_35, primals_36, buf153, 1216, 196, grid=grid(1216), stream=stream0)
        del primals_36
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 38, 1, 1), (38, 1, 1, 1))
        buf157 = buf156; del buf156  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_31.run(buf157, primals_124, 304, grid=grid(304), stream=stream0)
        del primals_124
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 152, 1, 1), (152, 1, 1, 1))
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf159, primals_126, 1216, grid=grid(1216), stream=stream0)
        del primals_126
        buf160 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_123], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_28.run(buf153, buf159, buf160, 238336, grid=grid(238336), stream=stream0)
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf162 = buf150; del buf150  # reuse
        buf163 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf165 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf161, primals_242, primals_243, buf162, buf163, buf165, primals_242, primals_243, 152, 1568, grid=grid(152), stream=stream0)
        del primals_242
        del primals_243
        buf166 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5, x_125, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_32.run(buf161, buf162, buf163, primals_37, primals_38, buf141, buf166, 238336, grid=grid(238336), stream=stream0)
        del primals_38
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf168 = buf163; del buf163  # reuse
        buf169 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf171 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf167, primals_245, primals_246, buf168, buf169, buf171, primals_245, primals_246, 152, 1568, grid=grid(152), stream=stream0)
        del primals_245
        del primals_246
        buf172 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134, x_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf167, buf168, buf169, primals_39, primals_40, buf172, 238336, grid=grid(238336), stream=stream0)
        del primals_40
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=19, bias=None)
        assert_size_stride(buf173, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf174 = buf169; del buf169  # reuse
        buf175 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf177 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf173, primals_248, primals_249, buf174, buf175, buf177, primals_248, primals_249, 152, 1568, grid=grid(152), stream=stream0)
        del primals_248
        del primals_249
        buf178 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((8, 152, 1, 1), (152, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf180 = reinterpret_tensor(buf179, (8, 152, 1, 1), (152, 1, 1, 1), 0); del buf179  # reuse
        # Source Nodes: [x_140, x_144, x_se_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_25.run(buf180, buf173, buf174, buf175, primals_41, primals_42, buf178, 1216, 196, grid=grid(1216), stream=stream0)
        del primals_42
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 38, 1, 1), (38, 1, 1, 1))
        buf182 = buf181; del buf181  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_31.run(buf182, primals_131, 304, grid=grid(304), stream=stream0)
        del primals_131
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 152, 1, 1), (152, 1, 1, 1))
        buf184 = buf183; del buf183  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf184, primals_133, 1216, grid=grid(1216), stream=stream0)
        del primals_133
        buf185 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_145], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_28.run(buf178, buf184, buf185, 238336, grid=grid(238336), stream=stream0)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf187 = buf175; del buf175  # reuse
        buf188 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf190 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_24.run(buf186, primals_251, primals_252, buf187, buf188, buf190, primals_251, primals_252, 152, 1568, grid=grid(152), stream=stream0)
        del primals_251
        del primals_252
        buf191 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_147, x_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_32.run(buf186, buf187, buf188, primals_43, primals_44, buf166, buf191, 238336, grid=grid(238336), stream=stream0)
        del buf188
        del primals_44
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 368, 14, 14), (72128, 196, 14, 1))
        buf193 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf196 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf192, primals_254, primals_255, buf193, buf194, buf196, primals_254, primals_255, 368, 1568, grid=grid(368), stream=stream0)
        del primals_254
        del primals_255
        buf197 = empty((8, 368, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_34.run(buf192, buf193, buf194, primals_45, primals_46, buf197, 577024, grid=grid(577024), stream=stream0)
        del primals_46
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_136, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf198, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf199 = buf194; del buf194  # reuse
        buf200 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf202 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf198, primals_257, primals_258, buf199, buf200, buf202, primals_257, primals_258, 368, 392, grid=grid(368), stream=stream0)
        del primals_257
        del primals_258
        buf203 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf204 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf205 = reinterpret_tensor(buf204, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf204  # reuse
        # Source Nodes: [x_163, x_167, x_se_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf205, buf198, buf199, buf200, primals_47, primals_48, buf203, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_48
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 38, 1, 1), (38, 1, 1, 1))
        buf207 = buf206; del buf206  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_31.run(buf207, primals_138, 304, grid=grid(304), stream=stream0)
        del primals_138
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 368, 1, 1), (368, 1, 1, 1))
        buf209 = buf208; del buf208  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf209, primals_140, 2944, grid=grid(2944), stream=stream0)
        del primals_140
        buf210 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_168], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf203, buf209, buf210, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf212 = buf200; del buf200  # reuse
        buf213 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf215 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf211, primals_260, primals_261, buf212, buf213, buf215, primals_260, primals_261, 368, 392, grid=grid(368), stream=stream0)
        del primals_260
        del primals_261
        # Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf191, primals_142, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf217 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf220 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf216, primals_263, primals_264, buf217, buf218, buf220, primals_263, primals_264, 368, 392, grid=grid(368), stream=stream0)
        del primals_263
        del primals_264
        buf221 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf222 = buf221; del buf221  # reuse
        # Source Nodes: [shortcut_7, x_170, x_176, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_39.run(buf222, buf211, buf212, buf213, primals_49, primals_50, buf216, buf217, buf218, primals_51, primals_52, 144256, grid=grid(144256), stream=stream0)
        del primals_50
        del primals_52
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf224 = buf218; del buf218  # reuse
        buf225 = buf213; del buf213  # reuse
        buf227 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf223, primals_266, primals_267, buf224, buf225, buf227, primals_266, primals_267, 368, 392, grid=grid(368), stream=stream0)
        del primals_266
        del primals_267
        buf228 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184, x_188], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf223, buf224, buf225, primals_53, primals_54, buf228, 144256, grid=grid(144256), stream=stream0)
        del primals_54
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf229, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf230 = buf225; del buf225  # reuse
        buf231 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf233 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf229, primals_269, primals_270, buf230, buf231, buf233, primals_269, primals_270, 368, 392, grid=grid(368), stream=stream0)
        del primals_269
        del primals_270
        buf234 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf235, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf235  # reuse
        # Source Nodes: [x_190, x_194, x_se_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf236, buf229, buf230, buf231, primals_55, primals_56, buf234, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_56
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 92, 1, 1), (92, 1, 1, 1))
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf238, primals_146, 736, grid=grid(736), stream=stream0)
        del primals_146
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 368, 1, 1), (368, 1, 1, 1))
        buf240 = buf239; del buf239  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf240, primals_148, 2944, grid=grid(2944), stream=stream0)
        del primals_148
        buf241 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_195], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf234, buf240, buf241, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf243 = buf231; del buf231  # reuse
        buf244 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf246 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf242, primals_272, primals_273, buf243, buf244, buf246, primals_272, primals_273, 368, 392, grid=grid(368), stream=stream0)
        del primals_272
        del primals_273
        buf247 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_8, x_197, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf242, buf243, buf244, primals_57, primals_58, buf222, buf247, 144256, grid=grid(144256), stream=stream0)
        del primals_58
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf249 = buf244; del buf244  # reuse
        buf250 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf252 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf248, primals_275, primals_276, buf249, buf250, buf252, primals_275, primals_276, 368, 392, grid=grid(368), stream=stream0)
        del primals_275
        del primals_276
        buf253 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206, x_210], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf248, buf249, buf250, primals_59, primals_60, buf253, 144256, grid=grid(144256), stream=stream0)
        del primals_60
        # Source Nodes: [x_211], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf254, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf255 = buf250; del buf250  # reuse
        buf256 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf258 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf254, primals_278, primals_279, buf255, buf256, buf258, primals_278, primals_279, 368, 392, grid=grid(368), stream=stream0)
        del primals_278
        del primals_279
        buf259 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf260 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf261 = reinterpret_tensor(buf260, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf260  # reuse
        # Source Nodes: [x_212, x_216, x_se_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf261, buf254, buf255, buf256, primals_61, primals_62, buf259, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_62
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 92, 1, 1), (92, 1, 1, 1))
        buf263 = buf262; del buf262  # reuse
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf263, primals_153, 736, grid=grid(736), stream=stream0)
        del primals_153
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 368, 1, 1), (368, 1, 1, 1))
        buf265 = buf264; del buf264  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf265, primals_155, 2944, grid=grid(2944), stream=stream0)
        del primals_155
        buf266 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8, x_217], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf259, buf265, buf266, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf268 = buf256; del buf256  # reuse
        buf269 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf271 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf267, primals_281, primals_282, buf268, buf269, buf271, primals_281, primals_282, 368, 392, grid=grid(368), stream=stream0)
        del primals_281
        del primals_282
        buf272 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_219, x_224], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf267, buf268, buf269, primals_63, primals_64, buf247, buf272, 144256, grid=grid(144256), stream=stream0)
        del primals_64
        # Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf274 = buf269; del buf269  # reuse
        buf275 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf277 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf273, primals_284, primals_285, buf274, buf275, buf277, primals_284, primals_285, 368, 392, grid=grid(368), stream=stream0)
        del primals_284
        del primals_285
        buf278 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf273, buf274, buf275, primals_65, primals_66, buf278, 144256, grid=grid(144256), stream=stream0)
        del primals_66
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_158, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf279, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf280 = buf275; del buf275  # reuse
        buf281 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf283 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf279, primals_287, primals_288, buf280, buf281, buf283, primals_287, primals_288, 368, 392, grid=grid(368), stream=stream0)
        del primals_287
        del primals_288
        buf284 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf285 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf286 = reinterpret_tensor(buf285, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf285  # reuse
        # Source Nodes: [x_234, x_238, x_se_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf286, buf279, buf280, buf281, primals_67, primals_68, buf284, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_68
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 92, 1, 1), (92, 1, 1, 1))
        buf288 = buf287; del buf287  # reuse
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf288, primals_160, 736, grid=grid(736), stream=stream0)
        del primals_160
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (8, 368, 1, 1), (368, 1, 1, 1))
        buf290 = buf289; del buf289  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf290, primals_162, 2944, grid=grid(2944), stream=stream0)
        del primals_162
        buf291 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9, x_239], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf284, buf290, buf291, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_240], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf293 = buf281; del buf281  # reuse
        buf294 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf296 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf292, primals_290, primals_291, buf293, buf294, buf296, primals_290, primals_291, 368, 392, grid=grid(368), stream=stream0)
        del primals_290
        del primals_291
        buf297 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10, x_241, x_246], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf292, buf293, buf294, primals_69, primals_70, buf272, buf297, 144256, grid=grid(144256), stream=stream0)
        del primals_70
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf299 = buf294; del buf294  # reuse
        buf300 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf302 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf298, primals_293, primals_294, buf299, buf300, buf302, primals_293, primals_294, 368, 392, grid=grid(368), stream=stream0)
        del primals_293
        del primals_294
        buf303 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250, x_254], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf298, buf299, buf300, primals_71, primals_72, buf303, 144256, grid=grid(144256), stream=stream0)
        del primals_72
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf304, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf305 = buf300; del buf300  # reuse
        buf306 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf308 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf304, primals_296, primals_297, buf305, buf306, buf308, primals_296, primals_297, 368, 392, grid=grid(368), stream=stream0)
        del primals_296
        del primals_297
        buf309 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf310 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf311 = reinterpret_tensor(buf310, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf310  # reuse
        # Source Nodes: [x_256, x_260, x_se_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf311, buf304, buf305, buf306, primals_73, primals_74, buf309, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_74
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 92, 1, 1), (92, 1, 1, 1))
        buf313 = buf312; del buf312  # reuse
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf313, primals_167, 736, grid=grid(736), stream=stream0)
        del primals_167
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 368, 1, 1), (368, 1, 1, 1))
        buf315 = buf314; del buf314  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf315, primals_169, 2944, grid=grid(2944), stream=stream0)
        del primals_169
        buf316 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10, x_261], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf309, buf315, buf316, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_262], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf318 = buf306; del buf306  # reuse
        buf319 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf321 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf317, primals_299, primals_300, buf318, buf319, buf321, primals_299, primals_300, 368, 392, grid=grid(368), stream=stream0)
        del primals_299
        del primals_300
        buf322 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_11, x_263, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf317, buf318, buf319, primals_75, primals_76, buf297, buf322, 144256, grid=grid(144256), stream=stream0)
        del primals_76
        # Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf324 = buf319; del buf319  # reuse
        buf325 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf327 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf323, primals_302, primals_303, buf324, buf325, buf327, primals_302, primals_303, 368, 392, grid=grid(368), stream=stream0)
        del primals_302
        del primals_303
        buf328 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272, x_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf323, buf324, buf325, primals_77, primals_78, buf328, 144256, grid=grid(144256), stream=stream0)
        del primals_78
        # Source Nodes: [x_277], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf329, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf330 = buf325; del buf325  # reuse
        buf331 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf333 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf329, primals_305, primals_306, buf330, buf331, buf333, primals_305, primals_306, 368, 392, grid=grid(368), stream=stream0)
        del primals_305
        del primals_306
        buf334 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf335 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf336 = reinterpret_tensor(buf335, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf335  # reuse
        # Source Nodes: [x_278, x_282, x_se_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf336, buf329, buf330, buf331, primals_79, primals_80, buf334, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_80
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 92, 1, 1), (92, 1, 1, 1))
        buf338 = buf337; del buf337  # reuse
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf338, primals_174, 736, grid=grid(736), stream=stream0)
        del primals_174
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf339 = extern_kernels.convolution(buf338, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf339, (8, 368, 1, 1), (368, 1, 1, 1))
        buf340 = buf339; del buf339  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf340, primals_176, 2944, grid=grid(2944), stream=stream0)
        del primals_176
        buf341 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11, x_283], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf334, buf340, buf341, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf343 = buf331; del buf331  # reuse
        buf344 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf346 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_285], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf342, primals_308, primals_309, buf343, buf344, buf346, primals_308, primals_309, 368, 392, grid=grid(368), stream=stream0)
        del primals_308
        del primals_309
        buf347 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_12, x_285, x_290], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_42.run(buf342, buf343, buf344, primals_81, primals_82, buf322, buf347, 144256, grid=grid(144256), stream=stream0)
        del primals_82
        # Source Nodes: [x_293], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf349 = buf344; del buf344  # reuse
        buf350 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf352 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf348, primals_311, primals_312, buf349, buf350, buf352, primals_311, primals_312, 368, 392, grid=grid(368), stream=stream0)
        del primals_311
        del primals_312
        buf353 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294, x_298], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf348, buf349, buf350, primals_83, primals_84, buf353, 144256, grid=grid(144256), stream=stream0)
        del primals_84
        # Source Nodes: [x_299], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_179, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=46, bias=None)
        assert_size_stride(buf354, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf355 = buf350; del buf350  # reuse
        buf356 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf358 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf354, primals_314, primals_315, buf355, buf356, buf358, primals_314, primals_315, 368, 392, grid=grid(368), stream=stream0)
        del primals_314
        del primals_315
        buf359 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        buf360 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf361 = reinterpret_tensor(buf360, (8, 368, 1, 1), (368, 1, 1, 1), 0); del buf360  # reuse
        # Source Nodes: [x_300, x_304, x_se_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_36.run(buf361, buf354, buf355, buf356, primals_85, primals_86, buf359, 2944, 49, grid=grid(2944), stream=stream0)
        del primals_86
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (8, 92, 1, 1), (92, 1, 1, 1))
        buf363 = buf362; del buf362  # reuse
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_41.run(buf363, primals_181, 736, grid=grid(736), stream=stream0)
        del primals_181
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 368, 1, 1), (368, 1, 1, 1))
        buf365 = buf364; del buf364  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf365, primals_183, 2944, grid=grid(2944), stream=stream0)
        del primals_183
        buf366 = empty((8, 368, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12, x_305], Original ATen: [aten.mul, aten.sigmoid]
        triton_poi_fused_mul_sigmoid_38.run(buf359, buf365, buf366, 144256, grid=grid(144256), stream=stream0)
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 368, 7, 7), (18032, 49, 7, 1))
        buf368 = buf356; del buf356  # reuse
        buf369 = empty_strided((1, 368, 1, 1), (368, 1, 368, 368), device='cuda', dtype=torch.float32)
        buf371 = empty((368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_35.run(buf367, primals_317, primals_318, buf368, buf369, buf371, primals_317, primals_318, 368, 392, grid=grid(368), stream=stream0)
        del primals_317
        del primals_318
        buf376 = empty((8, 368, 7, 7), device='cuda', dtype=torch.bool)
        buf373 = empty_strided((8, 368, 1, 1), (368, 1, 2944, 2944), device='cuda', dtype=torch.float32)
        buf374 = reinterpret_tensor(buf373, (8, 368), (368, 1), 0); del buf373  # reuse
        # Source Nodes: [x_307, x_312, x_315, x_318, x_320], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_43.run(buf374, buf367, buf368, buf369, primals_87, primals_88, buf347, buf376, 2944, 49, grid=grid(2944), stream=stream0)
        del buf369
        del primals_88
        buf375 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_186, buf374, reinterpret_tensor(primals_185, (368, 1000), (1, 368), 0), alpha=1, beta=1, out=buf375)
        del primals_186
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_187, primals_187, 1, grid=grid(1), stream=stream0)
        del primals_187
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_190, primals_190, 1, grid=grid(1), stream=stream0)
        del primals_190
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_193, primals_193, 1, grid=grid(1), stream=stream0)
        del primals_193
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_196, primals_196, 1, grid=grid(1), stream=stream0)
        del primals_196
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_199, primals_199, 1, grid=grid(1), stream=stream0)
        del primals_199
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_202, primals_202, 1, grid=grid(1), stream=stream0)
        del primals_202
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_205, primals_205, 1, grid=grid(1), stream=stream0)
        del primals_205
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_208, primals_208, 1, grid=grid(1), stream=stream0)
        del primals_208
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_211, primals_211, 1, grid=grid(1), stream=stream0)
        del primals_211
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_214, primals_214, 1, grid=grid(1), stream=stream0)
        del primals_214
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_217, primals_217, 1, grid=grid(1), stream=stream0)
        del primals_217
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_220, primals_220, 1, grid=grid(1), stream=stream0)
        del primals_220
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_223, primals_223, 1, grid=grid(1), stream=stream0)
        del primals_223
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_226, primals_226, 1, grid=grid(1), stream=stream0)
        del primals_226
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_229, primals_229, 1, grid=grid(1), stream=stream0)
        del primals_229
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_232, primals_232, 1, grid=grid(1), stream=stream0)
        del primals_232
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_235, primals_235, 1, grid=grid(1), stream=stream0)
        del primals_235
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_238, primals_238, 1, grid=grid(1), stream=stream0)
        del primals_238
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_241, primals_241, 1, grid=grid(1), stream=stream0)
        del primals_241
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_244, primals_244, 1, grid=grid(1), stream=stream0)
        del primals_244
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_247, primals_247, 1, grid=grid(1), stream=stream0)
        del primals_247
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_250, primals_250, 1, grid=grid(1), stream=stream0)
        del primals_250
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_253, primals_253, 1, grid=grid(1), stream=stream0)
        del primals_253
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_256, primals_256, 1, grid=grid(1), stream=stream0)
        del primals_256
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_259, primals_259, 1, grid=grid(1), stream=stream0)
        del primals_259
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_262, primals_262, 1, grid=grid(1), stream=stream0)
        del primals_262
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_265, primals_265, 1, grid=grid(1), stream=stream0)
        del primals_265
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_268, primals_268, 1, grid=grid(1), stream=stream0)
        del primals_268
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_271, primals_271, 1, grid=grid(1), stream=stream0)
        del primals_271
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_274, primals_274, 1, grid=grid(1), stream=stream0)
        del primals_274
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_277, primals_277, 1, grid=grid(1), stream=stream0)
        del primals_277
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_280, primals_280, 1, grid=grid(1), stream=stream0)
        del primals_280
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_283, primals_283, 1, grid=grid(1), stream=stream0)
        del primals_283
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_286, primals_286, 1, grid=grid(1), stream=stream0)
        del primals_286
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_289, primals_289, 1, grid=grid(1), stream=stream0)
        del primals_289
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_292, primals_292, 1, grid=grid(1), stream=stream0)
        del primals_292
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_295, primals_295, 1, grid=grid(1), stream=stream0)
        del primals_295
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_298, primals_298, 1, grid=grid(1), stream=stream0)
        del primals_298
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_301, primals_301, 1, grid=grid(1), stream=stream0)
        del primals_301
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_304, primals_304, 1, grid=grid(1), stream=stream0)
        del primals_304
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_307, primals_307, 1, grid=grid(1), stream=stream0)
        del primals_307
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_310, primals_310, 1, grid=grid(1), stream=stream0)
        del primals_310
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_313, primals_313, 1, grid=grid(1), stream=stream0)
        del primals_313
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(primals_316, primals_316, 1, grid=grid(1), stream=stream0)
        del primals_316
        return (buf375, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, buf0, buf7, buf8, buf9, buf16, buf17, buf18, buf25, buf26, buf28, buf30, buf32, buf33, buf34, buf41, buf42, buf49, buf51, buf52, buf59, buf60, buf61, buf65, buf66, buf68, buf70, buf72, buf73, buf74, buf78, buf79, buf83, buf85, buf86, buf90, buf91, buf92, buf96, buf97, buf99, buf101, buf103, buf104, buf105, buf109, buf110, buf114, buf116, buf117, buf121, buf122, buf123, buf127, buf128, buf130, buf132, buf134, buf135, buf136, buf140, buf141, buf142, buf146, buf147, buf148, buf152, buf153, buf155, buf157, buf159, buf160, buf161, buf165, buf166, buf167, buf171, buf172, buf173, buf177, buf178, buf180, buf182, buf184, buf185, buf186, buf190, buf191, buf192, buf196, buf197, buf198, buf202, buf203, buf205, buf207, buf209, buf210, buf211, buf215, buf216, buf220, buf222, buf223, buf227, buf228, buf229, buf233, buf234, buf236, buf238, buf240, buf241, buf242, buf246, buf247, buf248, buf252, buf253, buf254, buf258, buf259, buf261, buf263, buf265, buf266, buf267, buf271, buf272, buf273, buf277, buf278, buf279, buf283, buf284, buf286, buf288, buf290, buf291, buf292, buf296, buf297, buf298, buf302, buf303, buf304, buf308, buf309, buf311, buf313, buf315, buf316, buf317, buf321, buf322, buf323, buf327, buf328, buf329, buf333, buf334, buf336, buf338, buf340, buf341, buf342, buf346, buf347, buf348, buf352, buf353, buf354, buf358, buf359, buf361, buf363, buf365, buf366, buf367, buf371, buf374, reinterpret_tensor(primals_185, (1000, 368), (368, 1), 0), buf376, reinterpret_tensor(buf368, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf355, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf343, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf330, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf324, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf318, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf305, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf299, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf293, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf280, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf274, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf268, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf249, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf230, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf224, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf217, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf212, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf199, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf193, (1, 368, 1, 1), (368, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf174, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf168, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf162, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf149, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf143, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf137, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf124, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf118, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf106, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf87, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf80, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf62, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf56, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf46, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf38, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf22, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((24, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((8, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((24, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((56, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((6, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((56, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((56, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((56, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((14, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((152, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((152, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((152, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((38, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((152, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((152, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((38, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((368, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((368, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((368, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((92, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((92, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((368, 92, 1, 1), (92, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((368, 368, 1, 1), (368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1000, 368), (368, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_188 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_191 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_194 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_200 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_209 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_215 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_224 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_227 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_230 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_233 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_236 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_239 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_242 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_245 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_248 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_251 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_254 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_257 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_260 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_263 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_266 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_269 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_272 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_275 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_278 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_281 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_284 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_287 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_290 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_293 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_296 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_299 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_302 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_305 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_308 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_311 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_314 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_317 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('regnety_002', benchmark_compiled_module)
