
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
# Source Nodes: [l__mod___stem_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_1 => var_mean
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
# Source Nodes: [l__mod___stem_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
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
# Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___stem_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x => relu
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


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7w74abvron55suhpktnfssqm2xjyze3oayjm7qakswbixs7zr7.py
# Source Nodes: [l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_0_conv1_1 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6uyjpbxhhbct7kb6rvpawaamsya3wruczx2d7kbp4zgbgl7a6i.py
# Source Nodes: [l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_0_conv1_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eb/ceb6l64tqirjngafpmjhyauekszicj7sdenqqovxx2ufwzyitmxt.py
# Source Nodes: [cat_11, d1, l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_11 => cat
# d1 => relu_1
# l__mod___features_0_conv1_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    x2 = (xindex // 200704)
    x4 = xindex % 200704
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
    tl.store(out_ptr1 + (x4 + (401408*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykudcp3tggkqdxse3ijog745kcpqx6ozet6iok6vyvh3w6cmdco.py
# Source Nodes: [l__mod___features_0_conv2_1, l__mod___features_0_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_0_conv2_1 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# l__mod___features_0_conv2_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
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


# kernel path: /tmp/torchinductor_youkaichao/2w/c2wt73nvbaobchoydobafr2szehm4s7tmlc6fdvwja7gamb2bf5t.py
# Source Nodes: [l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_0_conv3_1 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dlpki6v4esmydm4bkbfncsqoc5rvm532tavdhs3fdn6vurxwu5.py
# Source Nodes: [l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_0_conv3_1 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sj/csjrq6sdgxeb24ckfpy7jfs5vrrj3p4dorjmefhkpw3o62rxsj6g.py
# Source Nodes: [cat_11, d2, l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_11 => cat
# d2 => relu_3
# l__mod___features_0_conv3_1 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
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
    tl.store(out_ptr1 + (x4 + (401408*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgz3xducotkpoglngpt6w6nztp2ukao3wn4nlwwrdzcmihbq5mr5.py
# Source Nodes: [d3, l__mod___features_0_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3 => relu_5
# l__mod___features_0_conv5_1 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqquadj5l2eitdhh7b4yh7bovleu4ggwpwjmb7koljkalxnixiuc.py
# Source Nodes: [cat_10, l__mod___features_0_conv6_1, out], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_10 => cat_1
# l__mod___features_0_conv6_1 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# out => relu_6
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    x2 = (xindex // 200704)
    x4 = xindex % 200704
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
    tl.store(out_ptr1 + (x4 + (602112*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2xcgmbfoxhbji3udotr27izypw42jefaofscu6ibw4nm4gohms.py
# Source Nodes: [cat_10, d2_1, l__mod___features_1_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_10 => cat_1
# d2_1 => relu_9
# l__mod___features_1_conv3_1 => add_46, add_49, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
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
    tl.store(out_ptr1 + (x4 + (602112*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhhcu2ikcz75l2semopv2hyldgdlq3nn47cq66uwfb6fhzbxapt.py
# Source Nodes: [d3_1, l__mod___features_1_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3_1 => relu_11
# l__mod___features_1_conv5_1 => add_56, add_59, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (602112*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catkindoyxqinzvd2ekdjj7on54fjzjnztfe5fxzx76qszl2jtbg.py
# Source Nodes: [l__mod___features_1_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_1_conv6_1 => var_mean_12
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/32/c32xl5favbhkhi5lgh3b542smvql46s7gtwjvkjs6aqo5ripvefs.py
# Source Nodes: [l__mod___features_1_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_1_conv6_1 => add_61, add_62, add_63, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3fyu5dn25b55bordiew7oz3oa2kmnhdeup7tc4o6adhi7iq4wp.py
# Source Nodes: [l__mod___features_1_conv6_1, l__mod___features_1_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_1_conv6_1 => add_61, add_64, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# l__mod___features_1_conv6_2 => relu_12
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
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


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmwgn2ybingrqwium5lcugjnywshsgtdghfzgc67xzkhvnczfpg.py
# Source Nodes: [l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_2_conv1_1 => add_66, add_67, add_68, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (112896*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yg/cygxqs4m7mgzgn4c56slyziqi7hlejpvtza4cxs2fu3cot2anmre.py
# Source Nodes: [cat_9, d1_2, l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_9 => cat_2
# d1_2 => relu_13
# l__mod___features_2_conv1_1 => add_66, add_69, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 144
    x2 = (xindex // 112896)
    x4 = xindex % 112896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x4 + (225792*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaoj7jf7hg5j3ywymsk63cwnuhjxa7ononeoxtiuxocwi3wrgvq.py
# Source Nodes: [l__mod___features_2_conv2_1, l__mod___features_2_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_2_conv2_1 => add_71, add_74, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# l__mod___features_2_conv2_2 => relu_14
triton_poi_fused__native_batch_norm_legit_functional_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 144
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wk/cwkc7hmm257n7fz5jsnioffhpw5vlcbhjy7odpihjllqrp2cmop6.py
# Source Nodes: [l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_2_conv3_1 => add_76, add_77, add_78, mul_106, mul_107, mul_108, mul_109, mul_110, rsqrt_15, squeeze_46, var_mean_15
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (56448*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/on/conupxrwp57e27h44ox3nw44vz4eif3dn5sltlm44sduuasxcaak.py
# Source Nodes: [cat_9, d2_2, l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_9 => cat_2
# d2_2 => relu_15
# l__mod___features_2_conv3_1 => add_76, add_79, mul_105, mul_111, rsqrt_15, sub_15, var_mean_15
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 72
    x2 = (xindex // 56448)
    x4 = xindex % 56448
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
    tl.store(out_ptr1 + (x4 + (225792*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwffv4pwjf6heap63kqkevzup5m4n3x5i3iaf6clbbzfp3clbnpr.py
# Source Nodes: [d3_2, l__mod___features_2_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3_2 => relu_17
# l__mod___features_2_conv5_1 => add_86, add_89, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 72
    x2 = (xindex // 56448)
    x4 = xindex % 56448
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (225792*x2)), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfbxyymljhptfqu7q3tglba5biltbdvg4x5l4rdrhqdr73cf6ds.py
# Source Nodes: [cat_8, l__mod___features_2_conv6_1, out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_8 => cat_3
# l__mod___features_2_conv6_1 => add_91, add_94, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# out_1 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 144
    x2 = (xindex // 112896)
    x4 = xindex % 112896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x4 + (338688*x2)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cby6jsq5uzrrb74x3cvjihyiwprvd5kvhxipeha3nmc3phxwe6ak.py
# Source Nodes: [cat_8, d2_3, l__mod___features_3_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_8 => cat_3
# d2_3 => relu_21
# l__mod___features_3_conv3_1 => add_106, add_109, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 72
    x2 = (xindex // 56448)
    x4 = xindex % 56448
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
    tl.store(out_ptr1 + (x4 + (338688*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ks/cksb2orft6mfzkjolbxpsy45maa6dl5zezzv2vgicyfajzerbxch.py
# Source Nodes: [d3_3, l__mod___features_3_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3_3 => relu_23
# l__mod___features_3_conv5_1 => add_116, add_119, mul_161, mul_167, rsqrt_23, sub_23, var_mean_23
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 72
    x2 = (xindex // 56448)
    x4 = xindex % 56448
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (338688*x2)), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbionrfnxwvswuif2anxqfqbnhggkbgffqoe3fxzl75cczgnbfnf.py
# Source Nodes: [l__mod___features_3_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_3_conv6_1 => add_121, add_122, add_123, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (225792*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eg/cegbr6l5ufy73hfxkjw2m66pibfrgp2gkni7vly5pxix3ny4q4wv.py
# Source Nodes: [l__mod___features_3_conv6_1, l__mod___features_3_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_3_conv6_1 => add_121, add_124, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# l__mod___features_3_conv6_2 => relu_24
triton_poi_fused__native_batch_norm_legit_functional_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5qgatm2ax5kc5v5bbcgexrjenldxd4iqmdf2c2aworfxobaofr.py
# Source Nodes: [l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_4_conv1_1 => add_126, add_127, add_128, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_25, squeeze_76, var_mean_25
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 304
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (59584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwwz4ymcnajxfyiztzrospd3exzikxh2yfdp5i5hzcjjyc2pg2b.py
# Source Nodes: [cat_7, d1_4, l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_7 => cat_4
# d1_4 => relu_25
# l__mod___features_4_conv1_1 => add_126, add_129, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 304
    x2 = (xindex // 59584)
    x4 = xindex % 59584
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
    tl.store(out_ptr1 + (x4 + (119168*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssjn7j3vjibpctexq3e2speaqp4tsak6wrtrn6be6sd7u4qzksm.py
# Source Nodes: [l__mod___features_4_conv2_1, l__mod___features_4_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_4_conv2_1 => add_131, add_134, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
# l__mod___features_4_conv2_2 => relu_26
triton_poi_fused__native_batch_norm_legit_functional_relu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 304
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


# kernel path: /tmp/torchinductor_youkaichao/3j/c3j5ya3mi2oreeurivl5yu7yslbcbisa57hxt74sfnxa4koioczj.py
# Source Nodes: [l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_4_conv3_1 => add_136, add_137, add_138, mul_190, mul_191, mul_192, mul_193, mul_194, rsqrt_27, squeeze_82, var_mean_27
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrbraj2ve2ml77indywnormygu74v533xbom55whw7nu5dqcycs.py
# Source Nodes: [cat_7, d2_4, l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_7 => cat_4
# d2_4 => relu_27
# l__mod___features_4_conv3_1 => add_136, add_139, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    x2 = (xindex // 29792)
    x4 = xindex % 29792
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
    tl.store(out_ptr1 + (x4 + (119168*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnojrjahqokcllkig6y7hawmhq62bakfgxoxythxmdatxiqksgdh.py
# Source Nodes: [d3_4, l__mod___features_4_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3_4 => relu_29
# l__mod___features_4_conv5_1 => add_146, add_149, mul_203, mul_209, rsqrt_29, sub_29, var_mean_29
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    x2 = (xindex // 29792)
    x4 = xindex % 29792
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (119168*x2)), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nsmcishn7fvwecda3s64i33dvrtdhtj5cqhdxkphgg5byvihdf.py
# Source Nodes: [cat_6, l__mod___features_4_conv6_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_6 => cat_5
# l__mod___features_4_conv6_1 => add_151, add_154, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# out_2 => relu_30
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 304
    x2 = (xindex // 59584)
    x4 = xindex % 59584
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
    tl.store(out_ptr1 + (x4 + (178752*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2qzq5j6g2su4hwmx5nwxp755wd7rw6ax5q2d2u6jxzpo34pwkt.py
# Source Nodes: [cat_6, d2_5, l__mod___features_5_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
# cat_6 => cat_5
# d2_5 => relu_33
# l__mod___features_5_conv3_1 => add_166, add_169, mul_231, mul_237, rsqrt_33, sub_33, var_mean_33
triton_poi_fused__native_batch_norm_legit_functional_cat_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_cat_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    x2 = (xindex // 29792)
    x4 = xindex % 29792
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
    tl.store(out_ptr1 + (x4 + (178752*x2)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7c/c7c6qh6z3vnzzjmbncq3bp3s2ww6poper7qbu4fwvxkamlf2ew2f.py
# Source Nodes: [d3_5, l__mod___features_5_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# d3_5 => relu_35
# l__mod___features_5_conv5_1 => add_176, add_179, mul_245, mul_251, rsqrt_35, sub_35, var_mean_35
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
    x2 = (xindex // 29792)
    x4 = xindex % 29792
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (178752*x2)), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcvcicuj3hhbu4lf363j5tyry2zf6rcyyk2fmbqk4jgj7dsrzwz.py
# Source Nodes: [l__mod___features_5_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___features_5_conv6_1 => add_181, add_182, add_183, mul_253, mul_254, mul_255, mul_256, mul_257, rsqrt_36, squeeze_109, var_mean_36
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (94080*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ap/capccgrm4tqaajehwaaszyenkqifarzbkqnnrscm6tdom7sbd2um.py
# Source Nodes: [l__mod___features_5_conv6_1, l__mod___features_5_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___features_5_conv6_1 => add_181, add_184, mul_252, mul_258, rsqrt_36, sub_36, var_mean_36
# l__mod___features_5_conv6_2 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 480
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


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhqwvmp5af3akiolnbye5tf2tx73qviuzg3iedticgwwzux4hgv.py
# Source Nodes: [l__mod___head_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_0_1 => add_186, add_187, add_188, mul_260, mul_261, mul_262, mul_263, mul_264, rsqrt_37, squeeze_112, var_mean_37
triton_per_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 960
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (47040*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oi/coip4qtrjxrzlsv5q64kptfhuc6sobcvz7w66qeyqhknivgnjlob.py
# Source Nodes: [l__mod___head_0_1, l__mod___head_0_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___head_0_1 => add_186, add_189, mul_259, mul_265, rsqrt_37, sub_37, var_mean_37
# l__mod___head_0_2 => relu_37
triton_poi_fused__native_batch_norm_legit_functional_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 960
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


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjircqwz4i4ifohejlwgadbs5pulwgoclvj6i6umn2fxj2n5com.py
# Source Nodes: [l__mod___head_1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_1_1 => add_191, add_192, add_193, mul_267, mul_268, mul_269, mul_270, mul_271, rsqrt_38, squeeze_115, var_mean_38
triton_per_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/o3/co3kqdeny3poaq2gnm5ghquu3e3qaikv277lhbji5nmwjivsubcf.py
# Source Nodes: [l__mod___head_1_1, l__mod___head_1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___head_1_1 => add_191, add_194, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
# l__mod___head_1_2 => relu_38
triton_poi_fused__native_batch_norm_legit_functional_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobgz5cairhexxttcuigqwhs2p72iwyypoji57rwxtxr45kleekn.py
# Source Nodes: [l__mod___head_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_2_1 => add_196, add_197, add_198, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_per_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = (rindex // 16)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0) + (20480*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0078740157480315
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


# kernel path: /tmp/torchinductor_youkaichao/de/cdecxdab3orsuudqfiium5esrdhfkoysa4uoqbcifrmxydyfpok2.py
# Source Nodes: [l__mod___head_2_1, l__mod___head_2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___head_2_1 => add_196, add_199, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# l__mod___head_2_2 => relu_39
triton_poi_fused__native_batch_norm_legit_functional_relu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16) % 1280
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ymflesksekfx7ifcvvwjku2cg6r7nazocwah75nzxiae26dyeb.py
# Source Nodes: [l__mod___head_3_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_3_1 => add_201, add_202, add_203, mul_281, mul_282, mul_283, mul_284, mul_285, rsqrt_40, squeeze_121, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = (rindex // 16)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0) + (16384*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0078740157480315
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtah55j4tmzxwmxmbafze4alzs3ox3ylofnavijeci3ldfgbowy.py
# Source Nodes: [l__mod___head_3_1, x_2, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# l__mod___head_3_1 => add_201, add_204, mul_280, mul_286, rsqrt_40, sub_40, var_mean_40
# x_2 => relu_40
# x_3 => mean
# x_5 => view
triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_view_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_view_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (16*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 16.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (r2 + (16*x3)), tmp16, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfl5qirtiksuryjktgtwtjoejq3vpe7cuemjat7qabbmubqeiiiy.py
# Source Nodes: [l__mod___stem_1], Original ATen: [aten.add]
# l__mod___stem_1 => add
triton_poi_fused_add_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_47', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (144, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_41, (144, ), (1, ))
    assert_size_stride(primals_42, (144, ), (1, ))
    assert_size_stride(primals_43, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_44, (144, ), (1, ))
    assert_size_stride(primals_45, (144, ), (1, ))
    assert_size_stride(primals_46, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_47, (72, ), (1, ))
    assert_size_stride(primals_48, (72, ), (1, ))
    assert_size_stride(primals_49, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_50, (144, ), (1, ))
    assert_size_stride(primals_51, (144, ), (1, ))
    assert_size_stride(primals_52, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_53, (72, ), (1, ))
    assert_size_stride(primals_54, (72, ), (1, ))
    assert_size_stride(primals_55, (144, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_56, (144, ), (1, ))
    assert_size_stride(primals_57, (144, ), (1, ))
    assert_size_stride(primals_58, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_59, (144, ), (1, ))
    assert_size_stride(primals_60, (144, ), (1, ))
    assert_size_stride(primals_61, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_62, (144, ), (1, ))
    assert_size_stride(primals_63, (144, ), (1, ))
    assert_size_stride(primals_64, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_65, (72, ), (1, ))
    assert_size_stride(primals_66, (72, ), (1, ))
    assert_size_stride(primals_67, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_68, (144, ), (1, ))
    assert_size_stride(primals_69, (144, ), (1, ))
    assert_size_stride(primals_70, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(primals_71, (72, ), (1, ))
    assert_size_stride(primals_72, (72, ), (1, ))
    assert_size_stride(primals_73, (288, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_74, (288, ), (1, ))
    assert_size_stride(primals_75, (288, ), (1, ))
    assert_size_stride(primals_76, (304, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(primals_77, (304, ), (1, ))
    assert_size_stride(primals_78, (304, ), (1, ))
    assert_size_stride(primals_79, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_80, (304, ), (1, ))
    assert_size_stride(primals_81, (304, ), (1, ))
    assert_size_stride(primals_82, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_83, (152, ), (1, ))
    assert_size_stride(primals_84, (152, ), (1, ))
    assert_size_stride(primals_85, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_86, (304, ), (1, ))
    assert_size_stride(primals_87, (304, ), (1, ))
    assert_size_stride(primals_88, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_89, (152, ), (1, ))
    assert_size_stride(primals_90, (152, ), (1, ))
    assert_size_stride(primals_91, (304, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_92, (304, ), (1, ))
    assert_size_stride(primals_93, (304, ), (1, ))
    assert_size_stride(primals_94, (304, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_95, (304, ), (1, ))
    assert_size_stride(primals_96, (304, ), (1, ))
    assert_size_stride(primals_97, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_98, (304, ), (1, ))
    assert_size_stride(primals_99, (304, ), (1, ))
    assert_size_stride(primals_100, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_101, (152, ), (1, ))
    assert_size_stride(primals_102, (152, ), (1, ))
    assert_size_stride(primals_103, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(primals_104, (304, ), (1, ))
    assert_size_stride(primals_105, (304, ), (1, ))
    assert_size_stride(primals_106, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(primals_107, (152, ), (1, ))
    assert_size_stride(primals_108, (152, ), (1, ))
    assert_size_stride(primals_109, (480, 912, 1, 1), (912, 1, 1, 1))
    assert_size_stride(primals_110, (480, ), (1, ))
    assert_size_stride(primals_111, (480, ), (1, ))
    assert_size_stride(primals_112, (960, 480, 3, 3), (4320, 9, 3, 1))
    assert_size_stride(primals_113, (960, ), (1, ))
    assert_size_stride(primals_114, (960, ), (1, ))
    assert_size_stride(primals_115, (1024, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1280, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_119, (1280, ), (1, ))
    assert_size_stride(primals_120, (1280, ), (1, ))
    assert_size_stride(primals_121, (1024, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_122, (1024, ), (1, ))
    assert_size_stride(primals_123, (1024, ), (1, ))
    assert_size_stride(primals_124, (1000, 1024), (1024, 1))
    assert_size_stride(primals_125, (1000, ), (1, ))
    assert_size_stride(primals_126, (32, ), (1, ))
    assert_size_stride(primals_127, (32, ), (1, ))
    assert_size_stride(primals_128, (), ())
    assert_size_stride(primals_129, (64, ), (1, ))
    assert_size_stride(primals_130, (64, ), (1, ))
    assert_size_stride(primals_131, (), ())
    assert_size_stride(primals_132, (64, ), (1, ))
    assert_size_stride(primals_133, (64, ), (1, ))
    assert_size_stride(primals_134, (), ())
    assert_size_stride(primals_135, (32, ), (1, ))
    assert_size_stride(primals_136, (32, ), (1, ))
    assert_size_stride(primals_137, (), ())
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (), ())
    assert_size_stride(primals_141, (32, ), (1, ))
    assert_size_stride(primals_142, (32, ), (1, ))
    assert_size_stride(primals_143, (), ())
    assert_size_stride(primals_144, (64, ), (1, ))
    assert_size_stride(primals_145, (64, ), (1, ))
    assert_size_stride(primals_146, (), ())
    assert_size_stride(primals_147, (64, ), (1, ))
    assert_size_stride(primals_148, (64, ), (1, ))
    assert_size_stride(primals_149, (), ())
    assert_size_stride(primals_150, (64, ), (1, ))
    assert_size_stride(primals_151, (64, ), (1, ))
    assert_size_stride(primals_152, (), ())
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (32, ), (1, ))
    assert_size_stride(primals_155, (), ())
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, ), (1, ))
    assert_size_stride(primals_158, (), ())
    assert_size_stride(primals_159, (32, ), (1, ))
    assert_size_stride(primals_160, (32, ), (1, ))
    assert_size_stride(primals_161, (), ())
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (144, ), (1, ))
    assert_size_stride(primals_166, (144, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (144, ), (1, ))
    assert_size_stride(primals_169, (144, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (72, ), (1, ))
    assert_size_stride(primals_172, (72, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (144, ), (1, ))
    assert_size_stride(primals_175, (144, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (72, ), (1, ))
    assert_size_stride(primals_178, (72, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (144, ), (1, ))
    assert_size_stride(primals_181, (144, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (144, ), (1, ))
    assert_size_stride(primals_184, (144, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (144, ), (1, ))
    assert_size_stride(primals_187, (144, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (72, ), (1, ))
    assert_size_stride(primals_190, (72, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (144, ), (1, ))
    assert_size_stride(primals_193, (144, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (288, ), (1, ))
    assert_size_stride(primals_199, (288, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (304, ), (1, ))
    assert_size_stride(primals_202, (304, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (304, ), (1, ))
    assert_size_stride(primals_205, (304, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (152, ), (1, ))
    assert_size_stride(primals_208, (152, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (304, ), (1, ))
    assert_size_stride(primals_211, (304, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (152, ), (1, ))
    assert_size_stride(primals_214, (152, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (304, ), (1, ))
    assert_size_stride(primals_217, (304, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (304, ), (1, ))
    assert_size_stride(primals_220, (304, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (304, ), (1, ))
    assert_size_stride(primals_223, (304, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (152, ), (1, ))
    assert_size_stride(primals_226, (152, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (304, ), (1, ))
    assert_size_stride(primals_229, (304, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (152, ), (1, ))
    assert_size_stride(primals_232, (152, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (960, ), (1, ))
    assert_size_stride(primals_238, (960, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (1280, ), (1, ))
    assert_size_stride(primals_244, (1280, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (1024, ), (1, ))
    assert_size_stride(primals_247, (1024, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_249, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf1 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 32, 1, 1, 13), (416, 13, 416, 416, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 416, 7720, grid=grid(416), stream=stream0)
        buf4 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf7 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_126, primals_127, buf4, buf5, buf7, primals_126, primals_127, 32, 13, grid=grid(32), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_126
        del primals_127
        buf8 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_2, primals_3, buf8, 3211264, grid=grid(3211264), stream=stream0)
        del primals_3
        # Source Nodes: [l__mod___features_0_conv1_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf10 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf9, buf10, buf11, buf12, 256, 6272, grid=grid(256), stream=stream0)
        buf13 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf16 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf10, buf11, buf12, primals_129, primals_130, buf13, buf14, buf16, primals_129, primals_130, 64, 4, grid=grid(64), stream=stream0)
        del primals_129
        del primals_130
        buf17 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf56 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf54 = reinterpret_tensor(buf56, (8, 64, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        # Source Nodes: [cat_11, d1, l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_5.run(buf9, buf13, buf14, primals_5, primals_6, buf17, buf54, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [l__mod___features_0_conv2_0], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf19 = buf12; del buf12  # reuse
        buf20 = buf11; del buf11  # reuse
        buf21 = buf10; del buf10  # reuse
        # Source Nodes: [l__mod___features_0_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf18, buf19, buf20, buf21, 256, 6272, grid=grid(256), stream=stream0)
        buf22 = buf14; del buf14  # reuse
        buf23 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf25 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf19, buf20, buf21, primals_132, primals_133, buf22, buf23, buf25, primals_132, primals_133, 64, 4, grid=grid(64), stream=stream0)
        del primals_132
        del primals_133
        buf26 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv2_1, l__mod___features_0_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf18, buf22, buf23, primals_8, primals_9, buf26, 1605632, grid=grid(1605632), stream=stream0)
        del primals_9
        # Source Nodes: [l__mod___features_0_conv3_0], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf28 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf27, buf28, buf29, buf30, 128, 6272, grid=grid(128), stream=stream0)
        buf31 = buf5; del buf5  # reuse
        buf32 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf34 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf28, buf29, buf30, primals_135, primals_136, buf31, buf32, buf34, primals_135, primals_136, 32, 4, grid=grid(32), stream=stream0)
        del primals_135
        del primals_136
        buf35 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf55 = reinterpret_tensor(buf56, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [cat_11, d2, l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_9.run(buf27, buf31, buf32, primals_11, primals_12, buf35, buf55, 802816, grid=grid(802816), stream=stream0)
        del primals_12
        # Source Nodes: [l__mod___features_0_conv4_0], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf37 = buf21; del buf21  # reuse
        buf38 = buf20; del buf20  # reuse
        buf39 = buf19; del buf19  # reuse
        # Source Nodes: [l__mod___features_0_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf36, buf37, buf38, buf39, 256, 6272, grid=grid(256), stream=stream0)
        buf40 = buf23; del buf23  # reuse
        buf41 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf43 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf37, buf38, buf39, primals_138, primals_139, buf40, buf41, buf43, primals_138, primals_139, 64, 4, grid=grid(64), stream=stream0)
        del primals_138
        del primals_139
        buf44 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv4_1, l__mod___features_0_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf36, buf40, buf41, primals_14, primals_15, buf44, 1605632, grid=grid(1605632), stream=stream0)
        del primals_15
        # Source Nodes: [l__mod___features_0_conv5_0], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf46 = buf30; del buf30  # reuse
        buf47 = buf29; del buf29  # reuse
        buf48 = buf28; del buf28  # reuse
        # Source Nodes: [l__mod___features_0_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf45, buf46, buf47, buf48, 128, 6272, grid=grid(128), stream=stream0)
        buf49 = buf32; del buf32  # reuse
        buf50 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf52 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf46, buf47, buf48, primals_141, primals_142, buf49, buf50, buf52, primals_141, primals_142, 32, 4, grid=grid(32), stream=stream0)
        del primals_141
        del primals_142
        buf53 = reinterpret_tensor(buf56, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        buf315 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3, l__mod___features_0_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_10.run(buf45, buf49, buf50, primals_17, primals_18, buf53, buf315, 802816, grid=grid(802816), stream=stream0)
        del primals_18
        # Source Nodes: [l__mod___features_0_conv6_0], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf58 = buf39; del buf39  # reuse
        buf59 = buf38; del buf38  # reuse
        buf60 = buf37; del buf37  # reuse
        # Source Nodes: [l__mod___features_0_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf57, buf58, buf59, buf60, 256, 6272, grid=grid(256), stream=stream0)
        buf61 = buf41; del buf41  # reuse
        buf62 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf64 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf58, buf59, buf60, primals_144, primals_145, buf61, buf62, buf64, primals_144, primals_145, 64, 4, grid=grid(64), stream=stream0)
        del primals_144
        del primals_145
        buf65 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf114 = empty((8, 192, 56, 56), device='cuda', dtype=torch.float32)
        buf113 = reinterpret_tensor(buf114, (8, 64, 56, 56), (602112, 3136, 56, 1), 401408)  # alias
        # Source Nodes: [cat_10, l__mod___features_0_conv6_1, out], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11.run(buf57, buf61, buf62, primals_20, primals_21, buf65, buf113, 1605632, grid=grid(1605632), stream=stream0)
        del primals_21
        # Source Nodes: [l__mod___features_1_conv1_0], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf67 = buf60; del buf60  # reuse
        buf68 = buf59; del buf59  # reuse
        buf69 = buf58; del buf58  # reuse
        # Source Nodes: [l__mod___features_1_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf66, buf67, buf68, buf69, 256, 6272, grid=grid(256), stream=stream0)
        buf70 = buf62; del buf62  # reuse
        buf71 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf73 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf67, buf68, buf69, primals_147, primals_148, buf70, buf71, buf73, primals_147, primals_148, 64, 4, grid=grid(64), stream=stream0)
        del primals_147
        del primals_148
        buf74 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf111 = reinterpret_tensor(buf114, (8, 64, 56, 56), (602112, 3136, 56, 1), 0)  # alias
        # Source Nodes: [cat_10, d1_1, l__mod___features_1_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_11.run(buf66, buf70, buf71, primals_23, primals_24, buf74, buf111, 1605632, grid=grid(1605632), stream=stream0)
        del primals_24
        # Source Nodes: [l__mod___features_1_conv2_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf76 = buf69; del buf69  # reuse
        buf77 = buf68; del buf68  # reuse
        buf78 = buf67; del buf67  # reuse
        # Source Nodes: [l__mod___features_1_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf75, buf76, buf77, buf78, 256, 6272, grid=grid(256), stream=stream0)
        buf79 = buf71; del buf71  # reuse
        buf80 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf82 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf76, buf77, buf78, primals_150, primals_151, buf79, buf80, buf82, primals_150, primals_151, 64, 4, grid=grid(64), stream=stream0)
        del primals_150
        del primals_151
        buf83 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv2_1, l__mod___features_1_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf75, buf79, buf80, primals_26, primals_27, buf83, 1605632, grid=grid(1605632), stream=stream0)
        del primals_27
        # Source Nodes: [l__mod___features_1_conv3_0], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf85 = buf48; del buf48  # reuse
        buf86 = buf47; del buf47  # reuse
        buf87 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___features_1_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf84, buf85, buf86, buf87, 128, 6272, grid=grid(128), stream=stream0)
        buf88 = buf50; del buf50  # reuse
        buf89 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf91 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf85, buf86, buf87, primals_153, primals_154, buf88, buf89, buf91, primals_153, primals_154, 32, 4, grid=grid(32), stream=stream0)
        del primals_153
        del primals_154
        buf92 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf112 = reinterpret_tensor(buf114, (8, 32, 56, 56), (602112, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [cat_10, d2_1, l__mod___features_1_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_12.run(buf84, buf88, buf89, primals_29, primals_30, buf92, buf112, 802816, grid=grid(802816), stream=stream0)
        del primals_30
        # Source Nodes: [l__mod___features_1_conv4_0], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf94 = buf78; del buf78  # reuse
        buf95 = buf77; del buf77  # reuse
        buf96 = buf76; del buf76  # reuse
        # Source Nodes: [l__mod___features_1_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf93, buf94, buf95, buf96, 256, 6272, grid=grid(256), stream=stream0)
        buf97 = buf80; del buf80  # reuse
        buf98 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf100 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf94, buf95, buf96, primals_156, primals_157, buf97, buf98, buf100, primals_156, primals_157, 64, 4, grid=grid(64), stream=stream0)
        del buf94
        del buf95
        del buf96
        del primals_156
        del primals_157
        buf101 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv4_1, l__mod___features_1_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf93, buf97, buf98, primals_32, primals_33, buf101, 1605632, grid=grid(1605632), stream=stream0)
        del buf98
        del primals_33
        # Source Nodes: [l__mod___features_1_conv5_0], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf103 = buf87; del buf87  # reuse
        buf104 = buf86; del buf86  # reuse
        buf105 = buf85; del buf85  # reuse
        # Source Nodes: [l__mod___features_1_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf102, buf103, buf104, buf105, 128, 6272, grid=grid(128), stream=stream0)
        buf106 = buf89; del buf89  # reuse
        buf107 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf109 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf103, buf104, buf105, primals_159, primals_160, buf106, buf107, buf109, primals_159, primals_160, 32, 4, grid=grid(32), stream=stream0)
        del primals_159
        del primals_160
        buf110 = reinterpret_tensor(buf114, (8, 32, 56, 56), (602112, 3136, 56, 1), 301056)  # alias
        buf314 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3_1, l__mod___features_1_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_13.run(buf102, buf106, buf107, primals_35, primals_36, buf110, buf314, 802816, grid=grid(802816), stream=stream0)
        del buf107
        del primals_36
        # Source Nodes: [l__mod___features_1_conv6_0], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf116 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf118 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf115, buf116, buf117, buf118, 512, 6272, grid=grid(512), stream=stream0)
        buf119 = reinterpret_tensor(buf105, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf105  # reuse
        buf120 = reinterpret_tensor(buf104, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf104  # reuse
        buf122 = reinterpret_tensor(buf103, (128, ), (1, ), 0); del buf103  # reuse
        # Source Nodes: [l__mod___features_1_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf116, buf117, buf118, primals_162, primals_163, buf119, buf120, buf122, primals_162, primals_163, 128, 4, grid=grid(128), stream=stream0)
        del buf116
        del buf117
        del buf118
        del primals_162
        del primals_163
        buf123 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_1_conv6_1, l__mod___features_1_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf115, buf119, buf120, primals_38, primals_39, buf123, 3211264, grid=grid(3211264), stream=stream0)
        del buf120
        del primals_39
        # Source Nodes: [l__mod___features_2_conv1_0], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_40, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf125 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf126 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf128 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf124, primals_165, primals_166, buf125, buf126, buf128, primals_165, primals_166, 144, 6272, grid=grid(144), stream=stream0)
        del primals_165
        del primals_166
        buf129 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        buf156 = empty((8, 288, 28, 28), device='cuda', dtype=torch.float32)
        buf154 = reinterpret_tensor(buf156, (8, 144, 28, 28), (225792, 784, 28, 1), 0)  # alias
        # Source Nodes: [cat_9, d1_2, l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_18.run(buf124, buf125, buf126, primals_41, primals_42, buf129, buf154, 903168, grid=grid(903168), stream=stream0)
        del primals_42
        # Source Nodes: [l__mod___features_2_conv2_0], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf131 = buf126; del buf126  # reuse
        buf132 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf134 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf130, primals_168, primals_169, buf131, buf132, buf134, primals_168, primals_169, 144, 6272, grid=grid(144), stream=stream0)
        del primals_168
        del primals_169
        buf135 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv2_1, l__mod___features_2_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf130, buf131, buf132, primals_44, primals_45, buf135, 903168, grid=grid(903168), stream=stream0)
        del primals_45
        # Source Nodes: [l__mod___features_2_conv3_0], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf137 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf138 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf140 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf136, primals_171, primals_172, buf137, buf138, buf140, primals_171, primals_172, 72, 6272, grid=grid(72), stream=stream0)
        del primals_171
        del primals_172
        buf141 = empty((8, 72, 28, 28), device='cuda', dtype=torch.float32)
        buf155 = reinterpret_tensor(buf156, (8, 72, 28, 28), (225792, 784, 28, 1), 112896)  # alias
        # Source Nodes: [cat_9, d2_2, l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_21.run(buf136, buf137, buf138, primals_47, primals_48, buf141, buf155, 451584, grid=grid(451584), stream=stream0)
        del primals_48
        # Source Nodes: [l__mod___features_2_conv4_0], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf143 = buf132; del buf132  # reuse
        buf144 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf146 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf142, primals_174, primals_175, buf143, buf144, buf146, primals_174, primals_175, 144, 6272, grid=grid(144), stream=stream0)
        del primals_174
        del primals_175
        buf147 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv4_1, l__mod___features_2_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf142, buf143, buf144, primals_50, primals_51, buf147, 903168, grid=grid(903168), stream=stream0)
        del primals_51
        # Source Nodes: [l__mod___features_2_conv5_0], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf149 = buf138; del buf138  # reuse
        buf150 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf152 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf148, primals_177, primals_178, buf149, buf150, buf152, primals_177, primals_178, 72, 6272, grid=grid(72), stream=stream0)
        del primals_177
        del primals_178
        buf153 = reinterpret_tensor(buf156, (8, 72, 28, 28), (225792, 784, 28, 1), 169344)  # alias
        buf313 = empty((8, 72, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3_2, l__mod___features_2_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_22.run(buf148, buf149, buf150, primals_53, primals_54, buf153, buf313, 451584, grid=grid(451584), stream=stream0)
        del primals_54
        # Source Nodes: [l__mod___features_2_conv6_0], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf158 = buf144; del buf144  # reuse
        buf159 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf161 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_2_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf157, primals_180, primals_181, buf158, buf159, buf161, primals_180, primals_181, 144, 6272, grid=grid(144), stream=stream0)
        del primals_180
        del primals_181
        buf162 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        buf196 = empty((8, 432, 28, 28), device='cuda', dtype=torch.float32)
        buf195 = reinterpret_tensor(buf196, (8, 144, 28, 28), (338688, 784, 28, 1), 225792)  # alias
        # Source Nodes: [cat_8, l__mod___features_2_conv6_1, out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_23.run(buf157, buf158, buf159, primals_56, primals_57, buf162, buf195, 903168, grid=grid(903168), stream=stream0)
        del primals_57
        # Source Nodes: [l__mod___features_3_conv1_0], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf164 = buf159; del buf159  # reuse
        buf165 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf167 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf163, primals_183, primals_184, buf164, buf165, buf167, primals_183, primals_184, 144, 6272, grid=grid(144), stream=stream0)
        del primals_183
        del primals_184
        buf168 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        buf193 = reinterpret_tensor(buf196, (8, 144, 28, 28), (338688, 784, 28, 1), 0)  # alias
        # Source Nodes: [cat_8, d1_3, l__mod___features_3_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_23.run(buf163, buf164, buf165, primals_59, primals_60, buf168, buf193, 903168, grid=grid(903168), stream=stream0)
        del primals_60
        # Source Nodes: [l__mod___features_3_conv2_0], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf170 = buf165; del buf165  # reuse
        buf171 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf173 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf169, primals_186, primals_187, buf170, buf171, buf173, primals_186, primals_187, 144, 6272, grid=grid(144), stream=stream0)
        del primals_186
        del primals_187
        buf174 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv2_1, l__mod___features_3_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf169, buf170, buf171, primals_62, primals_63, buf174, 903168, grid=grid(903168), stream=stream0)
        del primals_63
        # Source Nodes: [l__mod___features_3_conv3_0], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_64, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf176 = buf150; del buf150  # reuse
        buf177 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf179 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf175, primals_189, primals_190, buf176, buf177, buf179, primals_189, primals_190, 72, 6272, grid=grid(72), stream=stream0)
        del primals_189
        del primals_190
        buf180 = empty((8, 72, 28, 28), device='cuda', dtype=torch.float32)
        buf194 = reinterpret_tensor(buf196, (8, 72, 28, 28), (338688, 784, 28, 1), 112896)  # alias
        # Source Nodes: [cat_8, d2_3, l__mod___features_3_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_24.run(buf175, buf176, buf177, primals_65, primals_66, buf180, buf194, 451584, grid=grid(451584), stream=stream0)
        del primals_66
        # Source Nodes: [l__mod___features_3_conv4_0], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_67, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 144, 28, 28), (112896, 784, 28, 1))
        buf182 = buf171; del buf171  # reuse
        buf183 = empty_strided((1, 144, 1, 1), (144, 1, 144, 144), device='cuda', dtype=torch.float32)
        buf185 = empty((144, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf181, primals_192, primals_193, buf182, buf183, buf185, primals_192, primals_193, 144, 6272, grid=grid(144), stream=stream0)
        del primals_192
        del primals_193
        buf186 = empty((8, 144, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv4_1, l__mod___features_3_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf181, buf182, buf183, primals_68, primals_69, buf186, 903168, grid=grid(903168), stream=stream0)
        del buf183
        del primals_69
        # Source Nodes: [l__mod___features_3_conv5_0], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 72, 28, 28), (56448, 784, 28, 1))
        buf188 = buf177; del buf177  # reuse
        buf189 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cuda', dtype=torch.float32)
        buf191 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf187, primals_195, primals_196, buf188, buf189, buf191, primals_195, primals_196, 72, 6272, grid=grid(72), stream=stream0)
        del primals_195
        del primals_196
        buf192 = reinterpret_tensor(buf196, (8, 72, 28, 28), (338688, 784, 28, 1), 169344)  # alias
        buf312 = empty((8, 72, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3_3, l__mod___features_3_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_25.run(buf187, buf188, buf189, primals_71, primals_72, buf192, buf312, 451584, grid=grid(451584), stream=stream0)
        del buf189
        del primals_72
        # Source Nodes: [l__mod___features_3_conv6_0], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 288, 28, 28), (225792, 784, 28, 1))
        buf198 = empty_strided((1, 288, 1, 1), (288, 1, 288, 288), device='cuda', dtype=torch.float32)
        buf199 = empty_strided((1, 288, 1, 1), (288, 1, 288, 288), device='cuda', dtype=torch.float32)
        buf201 = empty((288, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf197, primals_198, primals_199, buf198, buf199, buf201, primals_198, primals_199, 288, 6272, grid=grid(288), stream=stream0)
        del primals_198
        del primals_199
        buf202 = empty((8, 288, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3_conv6_1, l__mod___features_3_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf197, buf198, buf199, primals_74, primals_75, buf202, 1806336, grid=grid(1806336), stream=stream0)
        del buf199
        del primals_75
        # Source Nodes: [l__mod___features_4_conv1_0], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_76, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf204 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf207 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf203, primals_201, primals_202, buf204, buf205, buf207, primals_201, primals_202, 304, 1568, grid=grid(304), stream=stream0)
        del primals_201
        del primals_202
        buf208 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        buf235 = empty((8, 608, 14, 14), device='cuda', dtype=torch.float32)
        buf233 = reinterpret_tensor(buf235, (8, 304, 14, 14), (119168, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_7, d1_4, l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_29.run(buf203, buf204, buf205, primals_77, primals_78, buf208, buf233, 476672, grid=grid(476672), stream=stream0)
        del primals_78
        # Source Nodes: [l__mod___features_4_conv2_0], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf210 = buf205; del buf205  # reuse
        buf211 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf213 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf209, primals_204, primals_205, buf210, buf211, buf213, primals_204, primals_205, 304, 1568, grid=grid(304), stream=stream0)
        del primals_204
        del primals_205
        buf214 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv2_1, l__mod___features_4_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf209, buf210, buf211, primals_80, primals_81, buf214, 476672, grid=grid(476672), stream=stream0)
        del primals_81
        # Source Nodes: [l__mod___features_4_conv3_0], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf216 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf217 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf219 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf215, primals_207, primals_208, buf216, buf217, buf219, primals_207, primals_208, 152, 1568, grid=grid(152), stream=stream0)
        del primals_207
        del primals_208
        buf220 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf234 = reinterpret_tensor(buf235, (8, 152, 14, 14), (119168, 196, 14, 1), 59584)  # alias
        # Source Nodes: [cat_7, d2_4, l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_32.run(buf215, buf216, buf217, primals_83, primals_84, buf220, buf234, 238336, grid=grid(238336), stream=stream0)
        del primals_84
        # Source Nodes: [l__mod___features_4_conv4_0], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf222 = buf211; del buf211  # reuse
        buf223 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf225 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf221, primals_210, primals_211, buf222, buf223, buf225, primals_210, primals_211, 304, 1568, grid=grid(304), stream=stream0)
        del primals_210
        del primals_211
        buf226 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv4_1, l__mod___features_4_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf221, buf222, buf223, primals_86, primals_87, buf226, 476672, grid=grid(476672), stream=stream0)
        del primals_87
        # Source Nodes: [l__mod___features_4_conv5_0], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf228 = buf217; del buf217  # reuse
        buf229 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf231 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf227, primals_213, primals_214, buf228, buf229, buf231, primals_213, primals_214, 152, 1568, grid=grid(152), stream=stream0)
        del primals_213
        del primals_214
        buf232 = reinterpret_tensor(buf235, (8, 152, 14, 14), (119168, 196, 14, 1), 89376)  # alias
        buf311 = empty((8, 152, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3_4, l__mod___features_4_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_33.run(buf227, buf228, buf229, primals_89, primals_90, buf232, buf311, 238336, grid=grid(238336), stream=stream0)
        del primals_90
        # Source Nodes: [l__mod___features_4_conv6_0], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf237 = buf223; del buf223  # reuse
        buf238 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf240 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_4_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf236, primals_216, primals_217, buf237, buf238, buf240, primals_216, primals_217, 304, 1568, grid=grid(304), stream=stream0)
        del primals_216
        del primals_217
        buf241 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        buf275 = empty((8, 912, 14, 14), device='cuda', dtype=torch.float32)
        buf274 = reinterpret_tensor(buf275, (8, 304, 14, 14), (178752, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_6, l__mod___features_4_conv6_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_34.run(buf236, buf237, buf238, primals_92, primals_93, buf241, buf274, 476672, grid=grid(476672), stream=stream0)
        del primals_93
        # Source Nodes: [l__mod___features_5_conv1_0], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf243 = buf238; del buf238  # reuse
        buf244 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf246 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf242, primals_219, primals_220, buf243, buf244, buf246, primals_219, primals_220, 304, 1568, grid=grid(304), stream=stream0)
        del primals_219
        del primals_220
        buf247 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        buf272 = reinterpret_tensor(buf275, (8, 304, 14, 14), (178752, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_6, d1_5, l__mod___features_5_conv1_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_34.run(buf242, buf243, buf244, primals_95, primals_96, buf247, buf272, 476672, grid=grid(476672), stream=stream0)
        del primals_96
        # Source Nodes: [l__mod___features_5_conv2_0], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(buf247, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf249 = buf244; del buf244  # reuse
        buf250 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf252 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf248, primals_222, primals_223, buf249, buf250, buf252, primals_222, primals_223, 304, 1568, grid=grid(304), stream=stream0)
        del primals_222
        del primals_223
        buf253 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv2_1, l__mod___features_5_conv2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf248, buf249, buf250, primals_98, primals_99, buf253, 476672, grid=grid(476672), stream=stream0)
        del primals_99
        # Source Nodes: [l__mod___features_5_conv3_0], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf255 = buf229; del buf229  # reuse
        buf256 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf258 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf254, primals_225, primals_226, buf255, buf256, buf258, primals_225, primals_226, 152, 1568, grid=grid(152), stream=stream0)
        del primals_225
        del primals_226
        buf259 = empty((8, 152, 14, 14), device='cuda', dtype=torch.float32)
        buf273 = reinterpret_tensor(buf275, (8, 152, 14, 14), (178752, 196, 14, 1), 59584)  # alias
        # Source Nodes: [cat_6, d2_5, l__mod___features_5_conv3_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_cat_relu_35.run(buf254, buf255, buf256, primals_101, primals_102, buf259, buf273, 238336, grid=grid(238336), stream=stream0)
        del primals_102
        # Source Nodes: [l__mod___features_5_conv4_0], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 304, 14, 14), (59584, 196, 14, 1))
        buf261 = buf250; del buf250  # reuse
        buf262 = empty_strided((1, 304, 1, 1), (304, 1, 304, 304), device='cuda', dtype=torch.float32)
        buf264 = empty((304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv4_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf260, primals_228, primals_229, buf261, buf262, buf264, primals_228, primals_229, 304, 1568, grid=grid(304), stream=stream0)
        del primals_228
        del primals_229
        buf265 = empty((8, 304, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv4_1, l__mod___features_5_conv4_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf260, buf261, buf262, primals_104, primals_105, buf265, 476672, grid=grid(476672), stream=stream0)
        del buf262
        del primals_105
        # Source Nodes: [l__mod___features_5_conv5_0], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 152, 14, 14), (29792, 196, 14, 1))
        buf267 = buf256; del buf256  # reuse
        buf268 = empty_strided((1, 152, 1, 1), (152, 1, 152, 152), device='cuda', dtype=torch.float32)
        buf270 = empty((152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf266, primals_231, primals_232, buf267, buf268, buf270, primals_231, primals_232, 152, 1568, grid=grid(152), stream=stream0)
        del primals_231
        del primals_232
        buf271 = reinterpret_tensor(buf275, (8, 152, 14, 14), (178752, 196, 14, 1), 89376)  # alias
        buf310 = empty((8, 152, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [d3_5, l__mod___features_5_conv5_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf266, buf267, buf268, primals_107, primals_108, buf271, buf310, 238336, grid=grid(238336), stream=stream0)
        del buf268
        del primals_108
        # Source Nodes: [l__mod___features_5_conv6_0], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 480, 14, 14), (94080, 196, 14, 1))
        buf277 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf278 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf280 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv6_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf276, primals_234, primals_235, buf277, buf278, buf280, primals_234, primals_235, 480, 1568, grid=grid(480), stream=stream0)
        del primals_234
        del primals_235
        buf281 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_5_conv6_1, l__mod___features_5_conv6_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf276, buf277, buf278, primals_110, primals_111, buf281, 752640, grid=grid(752640), stream=stream0)
        del buf278
        del primals_111
        # Source Nodes: [l__mod___head_0_0], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 960, 7, 7), (47040, 49, 7, 1))
        buf283 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf286 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_0_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_39.run(buf282, primals_237, primals_238, buf283, buf284, buf286, primals_237, primals_238, 960, 392, grid=grid(960), stream=stream0)
        del primals_237
        del primals_238
        buf287 = empty((8, 960, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_0_1, l__mod___head_0_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_40.run(buf282, buf283, buf284, primals_113, primals_114, buf287, 376320, grid=grid(376320), stream=stream0)
        del buf284
        del primals_114
        # Source Nodes: [l__mod___head_1_0], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf289 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf290 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf292 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_41.run(buf288, primals_240, primals_241, buf289, buf290, buf292, primals_240, primals_241, 1024, 392, grid=grid(1024), stream=stream0)
        del primals_240
        del primals_241
        buf293 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_1_1, l__mod___head_1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf288, buf289, buf290, primals_116, primals_117, buf293, 401408, grid=grid(401408), stream=stream0)
        del primals_117
        # Source Nodes: [l__mod___head_2_0], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 1280, 4, 4), (20480, 16, 4, 1))
        buf295 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf296 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf298 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_2_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_43.run(buf294, primals_243, primals_244, buf295, buf296, buf298, primals_243, primals_244, 1280, 128, grid=grid(1280), stream=stream0)
        del primals_243
        del primals_244
        buf299 = empty((8, 1280, 4, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_2_1, l__mod___head_2_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_44.run(buf294, buf295, buf296, primals_119, primals_120, buf299, 163840, grid=grid(163840), stream=stream0)
        del buf296
        del primals_120
        # Source Nodes: [l__mod___head_3_0], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 1024, 4, 4), (16384, 16, 4, 1))
        buf301 = buf290; del buf290  # reuse
        buf302 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf304 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_3_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf300, primals_246, primals_247, buf301, buf302, buf304, primals_246, primals_247, 1024, 128, grid=grid(1024), stream=stream0)
        del primals_246
        del primals_247
        buf309 = empty((8, 1024, 4, 4), device='cuda', dtype=torch.bool)
        buf306 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf307 = reinterpret_tensor(buf306, (8, 1024), (1024, 1), 0); del buf306  # reuse
        # Source Nodes: [l__mod___head_3_1, x_2, x_3, x_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_view_46.run(buf307, buf300, buf301, buf302, primals_122, primals_123, buf309, 8192, 16, grid=grid(8192), stream=stream0)
        del buf302
        del primals_123
        buf308 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf307, reinterpret_tensor(primals_124, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf308)
        del primals_125
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_128, primals_128, 1, grid=grid(1), stream=stream0)
        del primals_128
        # Source Nodes: [l__mod___features_0_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_131, primals_131, 1, grid=grid(1), stream=stream0)
        del primals_131
        # Source Nodes: [l__mod___features_0_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_134, primals_134, 1, grid=grid(1), stream=stream0)
        del primals_134
        # Source Nodes: [l__mod___features_0_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_137, primals_137, 1, grid=grid(1), stream=stream0)
        del primals_137
        # Source Nodes: [l__mod___features_0_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_140, primals_140, 1, grid=grid(1), stream=stream0)
        del primals_140
        # Source Nodes: [l__mod___features_0_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_143, primals_143, 1, grid=grid(1), stream=stream0)
        del primals_143
        # Source Nodes: [l__mod___features_0_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_146, primals_146, 1, grid=grid(1), stream=stream0)
        del primals_146
        # Source Nodes: [l__mod___features_1_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_149, primals_149, 1, grid=grid(1), stream=stream0)
        del primals_149
        # Source Nodes: [l__mod___features_1_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_152, primals_152, 1, grid=grid(1), stream=stream0)
        del primals_152
        # Source Nodes: [l__mod___features_1_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_155, primals_155, 1, grid=grid(1), stream=stream0)
        del primals_155
        # Source Nodes: [l__mod___features_1_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_158, primals_158, 1, grid=grid(1), stream=stream0)
        del primals_158
        # Source Nodes: [l__mod___features_1_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_161, primals_161, 1, grid=grid(1), stream=stream0)
        del primals_161
        # Source Nodes: [l__mod___features_1_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_164, primals_164, 1, grid=grid(1), stream=stream0)
        del primals_164
        # Source Nodes: [l__mod___features_2_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_167, primals_167, 1, grid=grid(1), stream=stream0)
        del primals_167
        # Source Nodes: [l__mod___features_2_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_170, primals_170, 1, grid=grid(1), stream=stream0)
        del primals_170
        # Source Nodes: [l__mod___features_2_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_173, primals_173, 1, grid=grid(1), stream=stream0)
        del primals_173
        # Source Nodes: [l__mod___features_2_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_176, primals_176, 1, grid=grid(1), stream=stream0)
        del primals_176
        # Source Nodes: [l__mod___features_2_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_179, primals_179, 1, grid=grid(1), stream=stream0)
        del primals_179
        # Source Nodes: [l__mod___features_2_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_182, primals_182, 1, grid=grid(1), stream=stream0)
        del primals_182
        # Source Nodes: [l__mod___features_3_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_185, primals_185, 1, grid=grid(1), stream=stream0)
        del primals_185
        # Source Nodes: [l__mod___features_3_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_188, primals_188, 1, grid=grid(1), stream=stream0)
        del primals_188
        # Source Nodes: [l__mod___features_3_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_191, primals_191, 1, grid=grid(1), stream=stream0)
        del primals_191
        # Source Nodes: [l__mod___features_3_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_194, primals_194, 1, grid=grid(1), stream=stream0)
        del primals_194
        # Source Nodes: [l__mod___features_3_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_197, primals_197, 1, grid=grid(1), stream=stream0)
        del primals_197
        # Source Nodes: [l__mod___features_3_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_200, primals_200, 1, grid=grid(1), stream=stream0)
        del primals_200
        # Source Nodes: [l__mod___features_4_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_203, primals_203, 1, grid=grid(1), stream=stream0)
        del primals_203
        # Source Nodes: [l__mod___features_4_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_206, primals_206, 1, grid=grid(1), stream=stream0)
        del primals_206
        # Source Nodes: [l__mod___features_4_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_209, primals_209, 1, grid=grid(1), stream=stream0)
        del primals_209
        # Source Nodes: [l__mod___features_4_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_212, primals_212, 1, grid=grid(1), stream=stream0)
        del primals_212
        # Source Nodes: [l__mod___features_4_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_215, primals_215, 1, grid=grid(1), stream=stream0)
        del primals_215
        # Source Nodes: [l__mod___features_4_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_218, primals_218, 1, grid=grid(1), stream=stream0)
        del primals_218
        # Source Nodes: [l__mod___features_5_conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_221, primals_221, 1, grid=grid(1), stream=stream0)
        del primals_221
        # Source Nodes: [l__mod___features_5_conv2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_224, primals_224, 1, grid=grid(1), stream=stream0)
        del primals_224
        # Source Nodes: [l__mod___features_5_conv3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_227, primals_227, 1, grid=grid(1), stream=stream0)
        del primals_227
        # Source Nodes: [l__mod___features_5_conv4_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_230, primals_230, 1, grid=grid(1), stream=stream0)
        del primals_230
        # Source Nodes: [l__mod___features_5_conv5_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_233, primals_233, 1, grid=grid(1), stream=stream0)
        del primals_233
        # Source Nodes: [l__mod___features_5_conv6_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_236, primals_236, 1, grid=grid(1), stream=stream0)
        del primals_236
        # Source Nodes: [l__mod___head_0_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_239, primals_239, 1, grid=grid(1), stream=stream0)
        del primals_239
        # Source Nodes: [l__mod___head_1_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_242, primals_242, 1, grid=grid(1), stream=stream0)
        del primals_242
        # Source Nodes: [l__mod___head_2_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_245, primals_245, 1, grid=grid(1), stream=stream0)
        del primals_245
        # Source Nodes: [l__mod___head_3_1], Original ATen: [aten.add]
        triton_poi_fused_add_47.run(primals_248, primals_248, 1, grid=grid(1), stream=stream0)
        del primals_248
        return (buf308, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_249, buf0, buf7, buf8, buf9, buf16, buf17, buf18, buf25, buf26, buf27, buf34, buf35, buf36, buf43, buf44, buf45, buf52, buf56, buf57, buf64, buf65, buf66, buf73, buf74, buf75, buf82, buf83, buf84, buf91, buf92, buf93, buf100, buf101, buf102, buf109, buf114, buf115, buf122, buf123, buf124, buf128, buf129, buf130, buf134, buf135, buf136, buf140, buf141, buf142, buf146, buf147, buf148, buf152, buf156, buf157, buf161, buf162, buf163, buf167, buf168, buf169, buf173, buf174, buf175, buf179, buf180, buf181, buf185, buf186, buf187, buf191, buf196, buf197, buf201, buf202, buf203, buf207, buf208, buf209, buf213, buf214, buf215, buf219, buf220, buf221, buf225, buf226, buf227, buf231, buf235, buf236, buf240, buf241, buf242, buf246, buf247, buf248, buf252, buf253, buf254, buf258, buf259, buf260, buf264, buf265, buf266, buf270, buf275, buf276, buf280, buf281, buf282, buf286, buf287, buf288, buf292, buf293, buf294, buf298, buf299, buf300, buf304, buf307, reinterpret_tensor(primals_124, (1000, 1024), (1024, 1), 0), buf309, reinterpret_tensor(buf301, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf295, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf289, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf283, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf277, (1, 480, 1, 1), (480, 1, 1, 1), 0), buf310, reinterpret_tensor(buf267, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf249, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf237, (1, 304, 1, 1), (304, 1, 1, 1), 0), buf311, reinterpret_tensor(buf228, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf222, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf216, (1, 152, 1, 1), (152, 1, 1, 1), 0), reinterpret_tensor(buf210, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf204, (1, 304, 1, 1), (304, 1, 1, 1), 0), reinterpret_tensor(buf198, (1, 288, 1, 1), (288, 1, 1, 1), 0), buf312, reinterpret_tensor(buf188, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf182, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf176, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf170, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf164, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf158, (1, 144, 1, 1), (144, 1, 1, 1), 0), buf313, reinterpret_tensor(buf149, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf143, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf137, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf131, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf125, (1, 144, 1, 1), (144, 1, 1, 1), 0), reinterpret_tensor(buf119, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf314, reinterpret_tensor(buf106, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf97, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf88, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf79, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf70, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf61, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf315, reinterpret_tensor(buf49, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf40, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf31, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf22, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((144, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((144, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((288, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((304, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((304, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((304, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((480, 912, 1, 1), (912, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((960, 480, 3, 3), (4320, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1024, 960, 3, 3), (8640, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1280, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_129 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_132 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_135 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_141 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_144 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_147 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_150 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_159 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('selecsls42b', benchmark_compiled_module)
