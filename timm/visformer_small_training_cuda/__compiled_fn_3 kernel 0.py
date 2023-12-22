
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


# kernel path: /tmp/torchinductor_youkaichao/ez/cezvw57ucvz5ryjcoksfj77nokbxyrsr2j2c7ovgaglvbf7lxhj6.py
# Source Nodes: [x_1], Original ATen: [aten.convolution]
# x_1 => convolution_1
triton_poi_fused_convolution_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdei5vq2el6wfxywglwufrnybkv7mtpk6kgcdhwnc2cxmpbnz7ed.py
# Source Nodes: [x_3], Original ATen: [aten._native_batch_norm_legit_functional]
# x_3 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_red_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3ufmkcqamniedqnq3tprptz6yqtv25pdjsevqqritpubiuid4c5.py
# Source Nodes: [add, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# add => add_10
# x_3 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_add_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_5', 'mutated_arg_names': []},
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
    x4 = xindex % 150528
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqtadzig5t75x6cqjr3kfz2qpylmldweqymydoy4azf7miyjpwx.py
# Source Nodes: [getattr_l__mod___stage1___0___norm2], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stage1___0___norm2 => add_12, add_15, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqxsimn4xsy3y6252gklj5oh6ndvyyaqgd4l7whseqpmd3siio4.py
# Source Nodes: [x_6], Original ATen: [aten.gelu]
# x_6 => add_16, erf, mul_21, mul_22, mul_23
triton_poi_fused_gelu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdom3q6p2d6zwuejmpdm6xpfr5f5xw2sgxungesyh4u2sdcwwrd.py
# Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___1___norm2 => add_20, add_21, add_22, mul_28, mul_29, mul_30, mul_31, mul_32, rsqrt_3, squeeze_10, var_mean_3
# x_12 => add_18
triton_red_fused__native_batch_norm_legit_functional_add_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_8', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tmp14 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = 6272.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.math.rsqrt(tmp10)
    tmp12 = 0.1
    tmp13 = tmp4 * tmp12
    tmp15 = 0.9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = 1.0001594642002871
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 * tmp12
    tmp22 = tmp21 * tmp15
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tl.store(out_ptr4 + (x0), tmp17, xmask)
    tl.store(out_ptr6 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkcdaeebc3ssvg6ztiufbbj4n77vfualvegs66yrxbwgp55hzoe.py
# Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___1___norm2 => add_20, add_23, mul_27, mul_33, rsqrt_3, sub_3, var_mean_3
# x_12 => add_18
triton_poi_fused__native_batch_norm_legit_functional_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_9', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 6272.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldyuoeah4ykpynqygvwrads2j2mk4pizdsq52krqmt3ahmml4x3.py
# Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___2___norm2 => add_28, add_29, add_30, mul_41, mul_42, mul_43, mul_44, mul_45, rsqrt_4, squeeze_13, var_mean_4
# x_12 => add_18
# x_20 => add_26
triton_red_fused__native_batch_norm_legit_functional_add_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 6272.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0001594642002871
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2ogncmgohtnujm5zlpta7ibvlvhmrtkiyd2vyqzb3ud2zxyxi5l.py
# Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___2___norm2 => add_28, add_31, mul_40, mul_46, rsqrt_4, sub_4, var_mean_4
# x_12 => add_18
# x_20 => add_26
triton_poi_fused__native_batch_norm_legit_functional_add_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 6272.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvmesbifzzl7r26hhi2v4623atsrlp6in43lacqhvhi7ofarqoo.py
# Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___3___norm2 => add_36, add_37, add_38, mul_54, mul_55, mul_56, mul_57, mul_58, rsqrt_5, squeeze_16, var_mean_5
# x_12 => add_18
# x_20 => add_26
# x_28 => add_34
triton_red_fused__native_batch_norm_legit_functional_add_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_12', 'mutated_arg_names': ['in_ptr4', 'in_ptr5', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
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
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = 6272.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = 0.1
    tmp17 = tmp8 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = 1.0001594642002871
    tmp23 = tmp12 * tmp22
    tmp24 = tmp23 * tmp16
    tmp26 = tmp25 * tmp19
    tmp27 = tmp24 + tmp26
    tl.store(out_ptr2 + (x0), tmp15, xmask)
    tl.store(out_ptr4 + (x0), tmp21, xmask)
    tl.store(out_ptr6 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcbi2dmptbpznuseoaz3zmonkc4w6eks6jjjhwtmex56rtcwjta.py
# Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage1___3___norm2 => add_36, add_39, mul_53, mul_59, rsqrt_5, sub_5, var_mean_5
# x_12 => add_18
# x_20 => add_26
# x_28 => add_34
triton_poi_fused__native_batch_norm_legit_functional_add_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_13', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 6272.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vgo6s6nlu3yw2pkmjgtyxyeyjur2gov25vx3o7yj64zzocb6jz.py
# Source Nodes: [x_12, x_20, x_28, x_36], Original ATen: [aten.add]
# x_12 => add_18
# x_20 => add_26
# x_28 => add_34
# x_36 => add_42
triton_poi_fused_add_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmorpsmgmuncvawzigdsuiwgjppwxmdfb6dad5ryvdtxjtsyubl4.py
# Source Nodes: [x_44, x_52, x_61], Original ATen: [aten.add]
# x_44 => add_50
# x_52 => add_58
# x_61 => add_66
triton_poi_fused_add_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmbprtyw6i4rcpjsgauchyjom5wgzuz6n5qfv4zxw2pp6he7pkx.py
# Source Nodes: [x_62], Original ATen: [aten.convolution]
# x_62 => convolution_23
triton_poi_fused_convolution_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tvrrenpmcscoirccwvqlgv5r5vcvq6gx3lfq655j23mvfi4ibk.py
# Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
# x_64 => add_68, add_69, add_70, mul_106, mul_107, mul_108, mul_109, mul_110, rsqrt_9, squeeze_28, var_mean_9
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cww6zrwyiazagh7gpi7g7izxtavxprjgy4mvvcl34mh6slddhmfx.py
# Source Nodes: [add_8, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# add_8 => add_72
# x_64 => add_68, add_71, mul_105, mul_111, rsqrt_9, sub_9, var_mean_9
triton_poi_fused__native_batch_norm_legit_functional_add_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_18', 'mutated_arg_names': []},
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
    x4 = xindex % 75264
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrhtd7rhsmbv4w5kz7dqujnmyrg2ooxrlnfaoaigkabc3kkx32e.py
# Source Nodes: [getattr_l__mod___stage2___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stage2___0___norm1 => add_74, add_77, mul_112, mul_118, rsqrt_10, sub_10, var_mean_10
triton_poi_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czle56beromuozctyyorjmqw3kmxc6utdmkuyev2vn4xelakrgzh.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_16
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9408
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 6
    y2 = (yindex // 1176)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x3) + (12544*y1) + (225792*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwxcmks3kgw7t4jnwaftc62nk6vbbpdmil76zd4cczkguttydcw.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_17
triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 75264
    x1 = (xindex // 75264)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (225792*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64kl4nxylj3qidstcyw6gdtasnh46h5frmb5t776wkekklk34y5.py
# Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn => mul_119
# attn_1 => amax, div, exp, sub_11, sum_1
# attn_2 => clone_18
triton_per_fused__softmax_clone_detach_mul_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
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
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (196*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (196*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/et/cets3avkamcdxliadkrfnagtrwc6fgabtxvhzxw3wopxxjn4gcp6.py
# Source Nodes: [x_67], Original ATen: [aten.clone]
# x_67 => clone_19
triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9408
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 6
    y2 = (yindex // 1176)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (150528 + y0 + (196*x3) + (12544*y1) + (225792*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgz677ak6d3i54fil7qkytwttwhweub5tkwez7quzu4omucrypf.py
# Source Nodes: [x_68], Original ATen: [aten._unsafe_view, aten.clone]
# x_68 => clone_20, view_7
triton_poi_fused__unsafe_view_clone_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + ((64*x2) + (12544*(y0 // 64)) + (75264*y1) + (y0 % 64)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rx/crxqde3uctnsossrdpbsdhph7k6eybgxpunjbxdp34466wohomlf.py
# Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___0___norm2 => add_80, add_81, add_82, mul_121, mul_122, mul_123, mul_124, mul_125, rsqrt_11, squeeze_34, var_mean_11
# x_71 => add_78
triton_red_fused__native_batch_norm_legit_functional_add_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_25', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
    tmp14 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = 1568.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.math.rsqrt(tmp10)
    tmp12 = 0.1
    tmp13 = tmp4 * tmp12
    tmp15 = 0.9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = 1.0006381620931717
    tmp19 = tmp8 * tmp18
    tmp20 = tmp19 * tmp12
    tmp22 = tmp21 * tmp15
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr2 + (x0), tmp11, xmask)
    tl.store(out_ptr4 + (x0), tmp17, xmask)
    tl.store(out_ptr6 + (x0), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clkun5kx2amdppmw37edxvtbgtlzdtvv6b4bfu537aqunmoboafk.py
# Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___0___norm2 => add_80, add_83, mul_120, mul_126, rsqrt_11, sub_12, var_mean_11
# x_71 => add_78
triton_poi_fused__native_batch_norm_legit_functional_add_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_26', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1568.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casnijzeemvpl2lj7e2jjjjqt75544vktitufrhvyrmwax6uwycz.py
# Source Nodes: [getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___1___norm1 => add_87, add_88, add_89, mul_131, mul_132, mul_133, mul_134, mul_135, rsqrt_12, squeeze_37, var_mean_12
# x_71 => add_78
# x_77 => add_85
triton_red_fused__native_batch_norm_legit_functional_add_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = 1568.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = 0.1
    tmp15 = tmp6 * tmp14
    tmp17 = 0.9
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = 1.0006381620931717
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21 * tmp14
    tmp24 = tmp23 * tmp17
    tmp25 = tmp22 + tmp24
    tl.store(out_ptr2 + (x0), tmp13, xmask)
    tl.store(out_ptr4 + (x0), tmp19, xmask)
    tl.store(out_ptr6 + (x0), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camxqawsfevdtof5uz4p6tsa6jwqeg5drgdcli7ctvyn72rs7d7r.py
# Source Nodes: [getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___1___norm1 => add_87, add_90, mul_130, mul_136, rsqrt_12, sub_13, var_mean_12
# x_71 => add_78
# x_77 => add_85
triton_poi_fused__native_batch_norm_legit_functional_add_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1568.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwmbd5gyz6n3cqr3gibgy425izsoddkyxtnq55xfxnleqkdupqc.py
# Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___1___norm2 => add_93, add_94, add_95, mul_139, mul_140, mul_141, mul_142, mul_143, rsqrt_13, squeeze_40, var_mean_13
# x_71 => add_78
# x_77 => add_85
# x_83 => add_91
triton_red_fused__native_batch_norm_legit_functional_add_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_29', 'mutated_arg_names': ['in_ptr4', 'in_ptr5', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
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
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = 1568.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = 0.1
    tmp17 = tmp8 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = 1.0006381620931717
    tmp23 = tmp12 * tmp22
    tmp24 = tmp23 * tmp16
    tmp26 = tmp25 * tmp19
    tmp27 = tmp24 + tmp26
    tl.store(out_ptr2 + (x0), tmp15, xmask)
    tl.store(out_ptr4 + (x0), tmp21, xmask)
    tl.store(out_ptr6 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czip44enzl5ju3yadexhqv2ivy2rl6xdk7kc5svclue7wn6xblxe.py
# Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage2___1___norm2 => add_93, add_96, mul_138, mul_144, rsqrt_13, sub_15, var_mean_13
# x_71 => add_78
# x_77 => add_85
# x_83 => add_91
triton_poi_fused__native_batch_norm_legit_functional_add_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_30', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1568.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqtbwpgf2rixp3wn3fbd2brg6czxylmhi4rwpnphznsbnvkx7is.py
# Source Nodes: [x_71, x_77, x_83, x_89], Original ATen: [aten.add]
# x_71 => add_78
# x_77 => add_85
# x_83 => add_91
# x_89 => add_98
triton_poi_fused_add_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6ysa7omthsv3bvfpmtq6qj4f2lmkakdxygitmrwkd4shffprv7.py
# Source Nodes: [x_101, x_107, x_114, x_95], Original ATen: [aten.add]
# x_101 => add_111
# x_107 => add_117
# x_114 => add_124
# x_95 => add_104
triton_poi_fused_add_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/criyrxbzxxorshk7gfzo26ezuydzswfqcxx35yi5toof2r6upykm.py
# Source Nodes: [x_115], Original ATen: [aten.convolution]
# x_115 => convolution_40
triton_poi_fused_convolution_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwujwyhlokrhpdrpx2si2lg5dm2v65kgxpdyphvatwqy77l43bk.py
# Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
# x_117 => add_126, add_127, add_128, mul_185, mul_186, mul_187, mul_188, mul_189, rsqrt_18, squeeze_55, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ah/cahwkvsnewyfcx65fcfqbjxsooghzwj7oacnjplq5umvtxfesjjn.py
# Source Nodes: [add_17, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# add_17 => add_130
# x_117 => add_126, add_129, mul_184, mul_190, rsqrt_18, sub_22, var_mean_18
triton_poi_fused__native_batch_norm_legit_functional_add_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_35', 'mutated_arg_names': []},
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
    x4 = xindex % 37632
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzwrknqhaoqjyip7tqw3lvxjo3oitu3ibdhaeu7vnuvphqvqqez.py
# Source Nodes: [getattr_l__mod___stage3___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stage3___0___norm1 => add_132, add_135, mul_191, mul_197, rsqrt_19, sub_23, var_mean_19
triton_poi_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd72aegfgpimxupzeawzhddwvls3xkjh34phrudr6tv23ycem7l7.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_49
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2352
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49) % 6
    y2 = (yindex // 294)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x3) + (6272*y1) + (112896*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6bpr2s46vlljoo6xz6dgcndyp6ecvkmhcdymmo5b2473gcamxtz.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_50
triton_poi_fused_clone_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 37632
    x1 = (xindex // 37632)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (112896*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ohy52bwoio35regogfdlri4n23dej27i6vmrrmmr5ms4i4iva7.py
# Source Nodes: [attn_12, attn_13, attn_14], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
# attn_12 => mul_198
# attn_13 => amax_4, div_4, exp_4, sub_24, sum_5
# attn_14 => clone_51
triton_per_fused__softmax_clone_detach_mul_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_clone_detach_mul_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2352
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
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (49*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (49*x0)), tmp13, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnyl6lbygwrnsyg2bf3m74zlgd3cjj2sjr5zeusy3zrgdo7wd6t.py
# Source Nodes: [x_120], Original ATen: [aten.clone]
# x_120 => clone_52
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2352
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49) % 6
    y2 = (yindex // 294)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (75264 + y0 + (49*x3) + (6272*y1) + (112896*y2)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (128*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gnlogpevcqvoalqmc67pukv5g2wkw6uym5whwqeg5iff5mxq6n.py
# Source Nodes: [x_121], Original ATen: [aten._unsafe_view, aten.clone]
# x_121 => clone_53, view_39
triton_poi_fused__unsafe_view_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + ((128*x2) + (6272*(y0 // 128)) + (37632*y1) + (y0 % 128)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckejvyp6ow3ebbybgxw4hvpb6wqw2u4nzaauoool6yf3ztkn7zsa.py
# Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___0___norm2 => add_138, add_139, add_140, mul_200, mul_201, mul_202, mul_203, mul_204, rsqrt_20, squeeze_61, var_mean_20
# x_124 => add_136
triton_per_fused__native_batch_norm_legit_functional_add_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_42', 'mutated_arg_names': ['in_ptr2', 'in_ptr3', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 392, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 392.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp24 = 0.1
    tmp25 = tmp12 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 1.0025575447570332
    tmp31 = tmp20 * tmp30
    tmp32 = tmp31 * tmp24
    tmp34 = tmp33 * tmp27
    tmp35 = tmp32 + tmp34
    tl.store(out_ptr2 + (x0), tmp23, xmask)
    tl.store(out_ptr4 + (x0), tmp29, xmask)
    tl.store(out_ptr6 + (x0), tmp35, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/cko7h4gjat6f2ndh4a2s3pcu5fut5j2jf4xwseyqf4syozt32alj.py
# Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___0___norm2 => add_138, add_141, mul_199, mul_205, rsqrt_20, sub_25, var_mean_20
# x_124 => add_136
triton_poi_fused__native_batch_norm_legit_functional_add_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_43', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 392.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqwnb6thcsa62rz7nkkjki76zmg6uamete2hjyddubozxywza4k.py
# Source Nodes: [x_126], Original ATen: [aten.gelu]
# x_126 => add_142, erf_18, mul_206, mul_207, mul_208
triton_poi_fused_gelu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = tl.math.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphhly72kvwzgl4btdref64fsmenuzmtmpplw5g4jvfd6guxfkkj.py
# Source Nodes: [getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___1___norm1 => add_145, add_146, add_147, mul_210, mul_211, mul_212, mul_213, mul_214, rsqrt_21, squeeze_64, var_mean_21
# x_124 => add_136
# x_130 => add_143
triton_per_fused__native_batch_norm_legit_functional_add_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_45', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 392, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 392.0
    tmp22 = tmp20 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = 0.1
    tmp27 = tmp14 * tmp26
    tmp29 = 0.9
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 + tmp30
    tmp32 = 1.0025575447570332
    tmp33 = tmp22 * tmp32
    tmp34 = tmp33 * tmp26
    tmp36 = tmp35 * tmp29
    tmp37 = tmp34 + tmp36
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp31, xmask)
    tl.store(out_ptr6 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2f7dgexc6yr3bzmonnoqxb23gznci52tqktgw42pda7qwljsfo.py
# Source Nodes: [getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___1___norm1 => add_145, add_148, mul_209, mul_215, rsqrt_21, sub_26, var_mean_21
# x_124 => add_136
# x_130 => add_143
triton_poi_fused__native_batch_norm_legit_functional_add_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 392.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7m3wcd7obdneigbhx66ojmpzte2x5l272wui2i3pk22bkvgsrj.py
# Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___1___norm2 => add_151, add_152, add_153, mul_218, mul_219, mul_220, mul_221, mul_222, rsqrt_22, squeeze_67, var_mean_22
# x_124 => add_136
# x_130 => add_143
# x_136 => add_149
triton_per_fused__native_batch_norm_legit_functional_add_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_47', 'mutated_arg_names': ['in_ptr4', 'in_ptr5', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 392, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = 392.0
    tmp24 = tmp22 / tmp23
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = tl.math.rsqrt(tmp26)
    tmp28 = 0.1
    tmp29 = tmp16 * tmp28
    tmp31 = 0.9
    tmp32 = tmp30 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = 1.0025575447570332
    tmp35 = tmp24 * tmp34
    tmp36 = tmp35 * tmp28
    tmp38 = tmp37 * tmp31
    tmp39 = tmp36 + tmp38
    tl.store(out_ptr2 + (x0), tmp27, xmask)
    tl.store(out_ptr4 + (x0), tmp33, xmask)
    tl.store(out_ptr6 + (x0), tmp39, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdg6jbirnsrrs4qceld3i55fwgw3yudqobcfnxsisylrccey5k7l.py
# Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# getattr_l__mod___stage3___1___norm2 => add_151, add_154, mul_217, mul_223, rsqrt_22, sub_28, var_mean_22
# x_124 => add_136
# x_130 => add_143
# x_136 => add_149
triton_poi_fused__native_batch_norm_legit_functional_add_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_48', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 392.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpnsq6be7uo7usmphfgfslpqjna64sdopl7ajfqiujuvcpwdoig.py
# Source Nodes: [x_124, x_130, x_136, x_142], Original ATen: [aten.add]
# x_124 => add_136
# x_130 => add_143
# x_136 => add_149
# x_142 => add_156
triton_poi_fused_add_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmunsbwzdmxmoupl4uspzvgjjiuzccxhv2cxodfqxtlx3fmzw5r.py
# Source Nodes: [x_169, x_170, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.view]
# x_169 => add_184, add_187, mul_263, mul_269, rsqrt_27, sub_35, var_mean_27
# x_170 => mean
# x_172 => view_64
triton_per_fused__native_batch_norm_legit_functional_mean_view_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_view_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
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


# kernel path: /tmp/torchinductor_youkaichao/rb/crb7g4cuopeyubgwyz4nphlee4elvvzvyrmz5m4vo2fgxv5jp2yo.py
# Source Nodes: [l__mod___stem_1], Original ATen: [aten.add]
# l__mod___stem_1 => add
triton_poi_fused_add_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_51', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206 = args
    args.clear()
    assert_size_stride(primals_1, (1, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(primals_2, (1, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(primals_3, (1, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(primals_4, (32, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (192, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (192, ), (1, ))
    assert_size_stride(primals_13, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_14, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_15, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_18, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_19, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_20, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_24, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_25, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_29, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_30, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_34, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_35, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_39, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_40, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, ), (1, ))
    assert_size_stride(primals_43, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_45, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_46, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_53, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_57, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (384, ), (1, ))
    assert_size_stride(primals_60, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_61, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_65, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_69, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_77, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_81, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_82, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_93, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_101, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_109, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_117, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (1000, 768), (768, 1))
    assert_size_stride(primals_121, (1000, ), (1, ))
    assert_size_stride(primals_122, (32, ), (1, ))
    assert_size_stride(primals_123, (32, ), (1, ))
    assert_size_stride(primals_124, (), ())
    assert_size_stride(primals_125, (192, ), (1, ))
    assert_size_stride(primals_126, (192, ), (1, ))
    assert_size_stride(primals_127, (), ())
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_130, (), ())
    assert_size_stride(primals_131, (192, ), (1, ))
    assert_size_stride(primals_132, (192, ), (1, ))
    assert_size_stride(primals_133, (), ())
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (), ())
    assert_size_stride(primals_137, (192, ), (1, ))
    assert_size_stride(primals_138, (192, ), (1, ))
    assert_size_stride(primals_139, (), ())
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (), ())
    assert_size_stride(primals_143, (192, ), (1, ))
    assert_size_stride(primals_144, (192, ), (1, ))
    assert_size_stride(primals_145, (), ())
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (), ())
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (), ())
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (), ())
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (), ())
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (), ())
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (), ())
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (), ())
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (), ())
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (), ())
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_178, (), ())
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (), ())
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (), ())
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_206, primals_4, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
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
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_122, primals_123, buf4, buf5, buf7, primals_122, primals_123, 32, 13, grid=grid(32), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_122
        del primals_123
        buf8 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_5, primals_6, buf8, 3211264, grid=grid(3211264), stream=stream0)
        del buf5
        del primals_6
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_7, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(buf10, primals_8, 1204224, grid=grid(1204224), stream=stream0)
        del primals_8
        buf11 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf14 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf10, primals_125, primals_126, buf11, buf12, buf14, primals_125, primals_126, 192, 6272, grid=grid(192), stream=stream0)
        del primals_125
        del primals_126
        buf15 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, x_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_5.run(buf10, buf11, buf12, primals_9, primals_10, primals_1, buf15, 1204224, grid=grid(1204224), stream=stream0)
        del primals_1
        del primals_10
        buf16 = buf12; del buf12  # reuse
        buf17 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf19 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___0___norm2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf15, primals_128, primals_129, buf16, buf17, buf19, primals_128, primals_129, 192, 6272, grid=grid(192), stream=stream0)
        del primals_128
        del primals_129
        buf20 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___0___norm2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_6.run(buf15, buf16, buf17, primals_11, primals_12, buf20, 1204224, grid=grid(1204224), stream=stream0)
        del primals_12
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf22 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf21, buf22, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf23, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf24 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf23, buf24, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf26 = buf17; del buf17  # reuse
        buf27 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf29 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_8.run(buf15, buf25, primals_131, primals_132, buf26, buf27, buf29, primals_131, primals_132, 192, 6272, grid=grid(192), stream=stream0)
        del primals_131
        del primals_132
        buf30 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___1___norm2, x_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_9.run(buf15, buf25, buf26, buf27, primals_16, primals_17, buf30, 1204224, grid=grid(1204224), stream=stream0)
        del primals_17
        # Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf32 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf31, buf32, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf33, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf34 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf33, buf34, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf36 = buf27; del buf27  # reuse
        buf37 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf39 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_10.run(buf15, buf25, buf35, primals_134, primals_135, buf36, buf37, buf39, primals_134, primals_135, 192, 6272, grid=grid(192), stream=stream0)
        del primals_134
        del primals_135
        buf40 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___2___norm2, x_12, x_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_11.run(buf15, buf25, buf35, buf36, buf37, primals_21, primals_22, buf40, 1204224, grid=grid(1204224), stream=stream0)
        del primals_22
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf42 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_22], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf41, buf42, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf43, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf44 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf43, buf44, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf46 = buf37; del buf37  # reuse
        buf47 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf49 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_12.run(buf15, buf25, buf35, buf45, primals_137, primals_138, buf46, buf47, buf49, primals_137, primals_138, 192, 6272, grid=grid(192), stream=stream0)
        del primals_137
        del primals_138
        buf50 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___3___norm2, x_12, x_20, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_13.run(buf15, buf25, buf35, buf45, buf46, buf47, primals_26, primals_27, buf50, 1204224, grid=grid(1204224), stream=stream0)
        del primals_27
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf52 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf51, buf52, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf53, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf54 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf53, buf54, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf56 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_20, x_28, x_36], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf15, buf25, buf35, buf45, buf55, buf56, 1204224, grid=grid(1204224), stream=stream0)
        buf57 = buf47; del buf47  # reuse
        buf58 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf60 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___4___norm2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf56, primals_140, primals_141, buf57, buf58, buf60, primals_140, primals_141, 192, 6272, grid=grid(192), stream=stream0)
        del primals_140
        del primals_141
        buf61 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___4___norm2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_6.run(buf56, buf57, buf58, primals_31, primals_32, buf61, 1204224, grid=grid(1204224), stream=stream0)
        del primals_32
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf63 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf62, buf63, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf64, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf65 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf64, buf65, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf67 = buf58; del buf58  # reuse
        buf68 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf70 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___5___norm2, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_8.run(buf56, buf66, primals_143, primals_144, buf67, buf68, buf70, primals_143, primals_144, 192, 6272, grid=grid(192), stream=stream0)
        del primals_143
        del primals_144
        buf71 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___5___norm2, x_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_9.run(buf56, buf66, buf67, buf68, primals_36, primals_37, buf71, 1204224, grid=grid(1204224), stream=stream0)
        del primals_37
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf73 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf72, buf73, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf74, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf75 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf74, buf75, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf77 = buf68; del buf68  # reuse
        buf78 = empty_strided((1, 192, 1, 1), (192, 1, 192, 192), device='cuda', dtype=torch.float32)
        buf80 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___6___norm2, x_44, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_10.run(buf56, buf66, buf76, primals_146, primals_147, buf77, buf78, buf80, primals_146, primals_147, 192, 6272, grid=grid(192), stream=stream0)
        del primals_146
        del primals_147
        buf81 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage1___6___norm2, x_44, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_11.run(buf56, buf66, buf76, buf77, buf78, primals_41, primals_42, buf81, 1204224, grid=grid(1204224), stream=stream0)
        del buf78
        del primals_42
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf83 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf82, buf83, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf84, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf85 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf84, buf85, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 192, 28, 28), (150528, 784, 28, 1))
        buf87 = buf86; del buf86  # reuse
        # Source Nodes: [x_44, x_52, x_61], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf87, buf56, buf66, buf76, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_46, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf89 = buf88; del buf88  # reuse
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf89, primals_47, 602112, grid=grid(602112), stream=stream0)
        del primals_47
        buf90 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf91 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf93 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf89, primals_149, primals_150, buf90, buf91, buf93, primals_149, primals_150, 384, 1568, grid=grid(384), stream=stream0)
        del primals_149
        del primals_150
        buf94 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_8, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_18.run(buf89, buf90, buf91, primals_48, primals_49, primals_2, buf94, 602112, grid=grid(602112), stream=stream0)
        del primals_2
        del primals_49
        buf95 = buf91; del buf91  # reuse
        buf96 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf98 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf94, primals_152, primals_153, buf95, buf96, buf98, primals_152, primals_153, 384, 1568, grid=grid(384), stream=stream0)
        del primals_152
        del primals_153
        buf99 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_19.run(buf94, buf95, buf96, primals_50, primals_51, buf99, 602112, grid=grid(602112), stream=stream0)
        del primals_51
        # Source Nodes: [getattr_l__mod___stage2___0___attn_qkv], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 1152, 14, 14), (225792, 196, 14, 1))
        buf101 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf100, buf101, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf102 = empty((8, 6, 64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf100, buf102, 602112, grid=grid(602112), stream=stream0)
        buf103 = empty((48, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf102, (48, 64, 196), (12544, 196, 1), 0), out=buf103)
        buf106 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        buf312 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, attn_2], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_22.run(buf103, buf106, buf312, 9408, 196, grid=grid(9408), stream=stream0)
        buf107 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf100, buf107, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf100
        buf108 = empty((48, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf107, (48, 196, 64), (12544, 64, 1), 0), out=buf108)
        buf109 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_24.run(buf108, buf109, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf111 = buf96; del buf96  # reuse
        buf112 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf114 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_25.run(buf94, buf110, primals_155, primals_156, buf111, buf112, buf114, primals_155, primals_156, 384, 1568, grid=grid(384), stream=stream0)
        del primals_155
        del primals_156
        buf115 = reinterpret_tensor(buf108, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf108  # reuse
        # Source Nodes: [getattr_l__mod___stage2___0___norm2, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_26.run(buf94, buf110, buf111, buf112, primals_54, primals_55, buf115, 602112, grid=grid(602112), stream=stream0)
        del primals_55
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf117 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf116, buf117, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf119 = buf112; del buf112  # reuse
        buf120 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf122 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_27.run(buf94, buf110, buf118, primals_158, primals_159, buf119, buf120, buf122, primals_158, primals_159, 384, 1568, grid=grid(384), stream=stream0)
        del primals_158
        del primals_159
        buf123 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___1___norm1, x_71, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_28.run(buf94, buf110, buf118, buf119, buf120, primals_58, primals_59, buf123, 602112, grid=grid(602112), stream=stream0)
        del primals_59
        # Source Nodes: [getattr_l__mod___stage2___1___attn_qkv], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 1152, 14, 14), (225792, 196, 14, 1))
        buf125 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf124, buf125, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf126 = empty((8, 6, 64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf124, buf126, 602112, grid=grid(602112), stream=stream0)
        buf127 = buf103; del buf103  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf126, (48, 64, 196), (12544, 196, 1), 0), out=buf127)
        buf130 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        buf311 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_3, attn_4, attn_5], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_22.run(buf127, buf130, buf311, 9408, 196, grid=grid(9408), stream=stream0)
        buf131 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf124, buf131, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf124
        buf132 = empty((48, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf131, (48, 196, 64), (12544, 64, 1), 0), out=buf132)
        buf133 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_24.run(buf132, buf133, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf135 = buf120; del buf120  # reuse
        buf136 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf138 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_29.run(buf94, buf110, buf118, buf134, primals_161, primals_162, buf135, buf136, buf138, primals_161, primals_162, 384, 1568, grid=grid(384), stream=stream0)
        del primals_161
        del primals_162
        buf139 = reinterpret_tensor(buf132, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf132  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___norm2, x_71, x_77, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_30.run(buf94, buf110, buf118, buf134, buf135, buf136, primals_62, primals_63, buf139, 602112, grid=grid(602112), stream=stream0)
        del primals_63
        # Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf141 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf140, buf141, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf143 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_77, x_83, x_89], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf94, buf110, buf118, buf134, buf142, buf143, 602112, grid=grid(602112), stream=stream0)
        buf144 = buf136; del buf136  # reuse
        buf145 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf147 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___2___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf143, primals_164, primals_165, buf144, buf145, buf147, primals_164, primals_165, 384, 1568, grid=grid(384), stream=stream0)
        del primals_164
        del primals_165
        buf148 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___2___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_19.run(buf143, buf144, buf145, primals_66, primals_67, buf148, 602112, grid=grid(602112), stream=stream0)
        del primals_67
        # Source Nodes: [getattr_l__mod___stage2___2___attn_qkv], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 1152, 14, 14), (225792, 196, 14, 1))
        buf150 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf149, buf150, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf151 = empty((8, 6, 64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf149, buf151, 602112, grid=grid(602112), stream=stream0)
        buf152 = buf127; del buf127  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf151, (48, 64, 196), (12544, 196, 1), 0), out=buf152)
        buf155 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        buf310 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, attn_8], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_22.run(buf152, buf155, buf310, 9408, 196, grid=grid(9408), stream=stream0)
        buf156 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf149, buf156, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf149
        buf157 = empty((48, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf156, (48, 196, 64), (12544, 64, 1), 0), out=buf157)
        buf158 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_24.run(buf157, buf158, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf160 = buf145; del buf145  # reuse
        buf161 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf163 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___2___norm2, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_25.run(buf143, buf159, primals_167, primals_168, buf160, buf161, buf163, primals_167, primals_168, 384, 1568, grid=grid(384), stream=stream0)
        del primals_167
        del primals_168
        buf164 = reinterpret_tensor(buf157, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf157  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___norm2, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_26.run(buf143, buf159, buf160, buf161, primals_70, primals_71, buf164, 602112, grid=grid(602112), stream=stream0)
        del primals_71
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf166 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf165, buf166, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf168 = buf161; del buf161  # reuse
        buf169 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf171 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___3___norm1, x_101, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_27.run(buf143, buf159, buf167, primals_170, primals_171, buf168, buf169, buf171, primals_170, primals_171, 384, 1568, grid=grid(384), stream=stream0)
        del primals_170
        del primals_171
        buf172 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___3___norm1, x_101, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_28.run(buf143, buf159, buf167, buf168, buf169, primals_74, primals_75, buf172, 602112, grid=grid(602112), stream=stream0)
        del primals_75
        # Source Nodes: [getattr_l__mod___stage2___3___attn_qkv], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 1152, 14, 14), (225792, 196, 14, 1))
        buf174 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf173, buf174, 9408, 64, grid=grid(9408, 64), stream=stream0)
        buf175 = empty((8, 6, 64, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf173, buf175, 602112, grid=grid(602112), stream=stream0)
        buf176 = buf152; del buf152  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (48, 196, 64), (12544, 64, 1), 0), reinterpret_tensor(buf175, (48, 64, 196), (12544, 196, 1), 0), out=buf176)
        buf179 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        buf309 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_11, attn_9], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_22.run(buf176, buf179, buf309, 9408, 196, grid=grid(9408), stream=stream0)
        del buf176
        buf180 = empty((8, 6, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf173, buf180, 9408, 64, grid=grid(9408, 64), stream=stream0)
        del buf173
        buf181 = empty((48, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf179, (48, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf180, (48, 196, 64), (12544, 64, 1), 0), out=buf181)
        buf182 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_24.run(buf181, buf182, 3072, 196, grid=grid(3072, 196), stream=stream0)
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf184 = buf169; del buf169  # reuse
        buf185 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf187 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___3___norm2, x_101, x_107, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_red_fused__native_batch_norm_legit_functional_add_29.run(buf143, buf159, buf167, buf183, primals_173, primals_174, buf184, buf185, buf187, primals_173, primals_174, 384, 1568, grid=grid(384), stream=stream0)
        del primals_173
        del primals_174
        buf188 = reinterpret_tensor(buf181, (8, 384, 14, 14), (75264, 196, 14, 1), 0); del buf181  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___norm2, x_101, x_107, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_30.run(buf143, buf159, buf167, buf183, buf184, buf185, primals_78, primals_79, buf188, 602112, grid=grid(602112), stream=stream0)
        del buf185
        del primals_79
        # Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf190 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf189, buf190, 2408448, grid=grid(2408448), stream=stream0)
        # Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf192 = buf191; del buf191  # reuse
        # Source Nodes: [x_101, x_107, x_114, x_95], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf192, buf143, buf159, buf167, buf183, 602112, grid=grid(602112), stream=stream0)
        del buf143
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf194 = buf193; del buf193  # reuse
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_33.run(buf194, primals_83, 301056, grid=grid(301056), stream=stream0)
        del primals_83
        buf195 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf198 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf194, primals_176, primals_177, buf195, buf196, buf198, primals_176, primals_177, 768, 392, grid=grid(768), stream=stream0)
        del primals_176
        del primals_177
        buf199 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_17, x_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_35.run(buf194, buf195, buf196, primals_84, primals_85, primals_3, buf199, 301056, grid=grid(301056), stream=stream0)
        del primals_3
        del primals_85
        buf200 = buf196; del buf196  # reuse
        buf201 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf203 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf199, primals_179, primals_180, buf200, buf201, buf203, primals_179, primals_180, 768, 392, grid=grid(768), stream=stream0)
        del primals_179
        del primals_180
        buf204 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_36.run(buf199, buf200, buf201, primals_86, primals_87, buf204, 301056, grid=grid(301056), stream=stream0)
        del primals_87
        # Source Nodes: [getattr_l__mod___stage3___0___attn_qkv], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 2304, 7, 7), (112896, 49, 7, 1))
        buf206 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf205, buf206, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf207 = empty((8, 6, 128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf205, buf207, 301056, grid=grid(301056), stream=stream0)
        buf208 = empty((48, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf206, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf207, (48, 128, 49), (6272, 49, 1), 0), out=buf208)
        buf211 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        buf308 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, attn_14], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_39.run(buf208, buf211, buf308, 2352, 49, grid=grid(2352), stream=stream0)
        buf212 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf205, buf212, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf205
        buf213 = empty((48, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf212, (48, 49, 128), (6272, 128, 1), 0), out=buf213)
        buf214 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_121], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_41.run(buf213, buf214, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf216 = buf201; del buf201  # reuse
        buf217 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf219 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_42.run(buf199, buf215, primals_182, primals_183, buf216, buf217, buf219, primals_182, primals_183, 768, 392, grid=grid(768), stream=stream0)
        del primals_182
        del primals_183
        buf220 = reinterpret_tensor(buf213, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf213  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___norm2, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_43.run(buf199, buf215, buf216, buf217, primals_90, primals_91, buf220, 301056, grid=grid(301056), stream=stream0)
        del primals_91
        # Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf222 = reinterpret_tensor(buf56, (8, 3072, 7, 7), (150528, 49, 7, 1), 0); del buf56  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf221, buf222, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf224 = buf217; del buf217  # reuse
        buf225 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_45.run(buf199, buf215, buf223, primals_185, primals_186, buf224, buf225, buf227, primals_185, primals_186, 768, 392, grid=grid(768), stream=stream0)
        del primals_185
        del primals_186
        buf228 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___1___norm1, x_124, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_46.run(buf199, buf215, buf223, buf224, buf225, primals_94, primals_95, buf228, 301056, grid=grid(301056), stream=stream0)
        del primals_95
        # Source Nodes: [getattr_l__mod___stage3___1___attn_qkv], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (8, 2304, 7, 7), (112896, 49, 7, 1))
        buf230 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf229, buf230, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf231 = empty((8, 6, 128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf229, buf231, 301056, grid=grid(301056), stream=stream0)
        buf232 = buf208; del buf208  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf230, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf231, (48, 128, 49), (6272, 49, 1), 0), out=buf232)
        buf235 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        buf307 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_15, attn_16, attn_17], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_39.run(buf232, buf235, buf307, 2352, 49, grid=grid(2352), stream=stream0)
        buf236 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf229, buf236, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf229
        buf237 = empty((48, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf236, (48, 49, 128), (6272, 128, 1), 0), out=buf237)
        buf238 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_41.run(buf237, buf238, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf240 = buf225; del buf225  # reuse
        buf241 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf243 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_47.run(buf199, buf215, buf223, buf239, primals_188, primals_189, buf240, buf241, buf243, primals_188, primals_189, 768, 392, grid=grid(768), stream=stream0)
        del primals_188
        del primals_189
        buf244 = reinterpret_tensor(buf237, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf237  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___norm2, x_124, x_130, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf199, buf215, buf223, buf239, buf240, buf241, primals_98, primals_99, buf244, 301056, grid=grid(301056), stream=stream0)
        del primals_99
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf246 = empty((8, 3072, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf245, buf246, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf248 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124, x_130, x_136, x_142], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(buf199, buf215, buf223, buf239, buf247, buf248, 301056, grid=grid(301056), stream=stream0)
        buf249 = buf241; del buf241  # reuse
        buf250 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf252 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___2___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf248, primals_191, primals_192, buf249, buf250, buf252, primals_191, primals_192, 768, 392, grid=grid(768), stream=stream0)
        del primals_191
        del primals_192
        buf253 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___2___norm1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_36.run(buf248, buf249, buf250, primals_102, primals_103, buf253, 301056, grid=grid(301056), stream=stream0)
        del primals_103
        # Source Nodes: [getattr_l__mod___stage3___2___attn_qkv], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 2304, 7, 7), (112896, 49, 7, 1))
        buf255 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf254, buf255, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf256 = empty((8, 6, 128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf254, buf256, 301056, grid=grid(301056), stream=stream0)
        buf257 = buf232; del buf232  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf255, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf256, (48, 128, 49), (6272, 49, 1), 0), out=buf257)
        buf260 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        buf306 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19, attn_20], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_39.run(buf257, buf260, buf306, 2352, 49, grid=grid(2352), stream=stream0)
        buf261 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf254, buf261, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf254
        buf262 = empty((48, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf261, (48, 49, 128), (6272, 128, 1), 0), out=buf262)
        buf263 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_41.run(buf262, buf263, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf265 = buf250; del buf250  # reuse
        buf266 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf268 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___2___norm2, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_42.run(buf248, buf264, primals_194, primals_195, buf265, buf266, buf268, primals_194, primals_195, 768, 392, grid=grid(768), stream=stream0)
        del primals_194
        del primals_195
        buf269 = reinterpret_tensor(buf262, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf262  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___norm2, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_43.run(buf248, buf264, buf265, buf266, primals_106, primals_107, buf269, 301056, grid=grid(301056), stream=stream0)
        del primals_107
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf271 = empty((8, 3072, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf270, buf271, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_152], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf273 = buf266; del buf266  # reuse
        buf274 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf276 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___3___norm1, x_148, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_45.run(buf248, buf264, buf272, primals_197, primals_198, buf273, buf274, buf276, primals_197, primals_198, 768, 392, grid=grid(768), stream=stream0)
        del primals_197
        del primals_198
        buf277 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___3___norm1, x_148, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_46.run(buf248, buf264, buf272, buf273, buf274, primals_110, primals_111, buf277, 301056, grid=grid(301056), stream=stream0)
        del primals_111
        # Source Nodes: [getattr_l__mod___stage3___3___attn_qkv], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 2304, 7, 7), (112896, 49, 7, 1))
        buf279 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf278, buf279, 2352, 128, grid=grid(2352, 128), stream=stream0)
        buf280 = empty((8, 6, 128, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_38.run(buf278, buf280, 301056, grid=grid(301056), stream=stream0)
        buf281 = buf257; del buf257  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (48, 49, 128), (6272, 128, 1), 0), reinterpret_tensor(buf280, (48, 128, 49), (6272, 49, 1), 0), out=buf281)
        buf284 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        buf305 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_21, attn_22, attn_23], Original ATen: [aten._softmax, aten.clone, aten.detach, aten.mul]
        triton_per_fused__softmax_clone_detach_mul_39.run(buf281, buf284, buf305, 2352, 49, grid=grid(2352), stream=stream0)
        del buf281
        buf285 = empty((8, 6, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf278, buf285, 2352, 128, grid=grid(2352, 128), stream=stream0)
        del buf278
        buf286 = empty((48, 49, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (48, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf285, (48, 49, 128), (6272, 128, 1), 0), out=buf286)
        buf287 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_41.run(buf286, buf287, 6144, 49, grid=grid(6144, 49), stream=stream0)
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf289 = buf274; del buf274  # reuse
        buf290 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf292 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___3___norm2, x_148, x_154, x_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_per_fused__native_batch_norm_legit_functional_add_47.run(buf248, buf264, buf272, buf288, primals_200, primals_201, buf289, buf290, buf292, primals_200, primals_201, 768, 392, grid=grid(768), stream=stream0)
        del primals_200
        del primals_201
        buf293 = reinterpret_tensor(buf286, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf286  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___norm2, x_148, x_154, x_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_48.run(buf248, buf264, buf272, buf288, buf289, buf290, primals_114, primals_115, buf293, 301056, grid=grid(301056), stream=stream0)
        del primals_115
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 3072, 7, 7), (150528, 49, 7, 1))
        buf295 = empty((8, 3072, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_44.run(buf294, buf295, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 768, 7, 7), (37632, 49, 7, 1))
        buf297 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148, x_154, x_160, x_167], Original ATen: [aten.add]
        triton_poi_fused_add_49.run(buf248, buf264, buf272, buf288, buf296, buf297, 301056, grid=grid(301056), stream=stream0)
        del buf248
        buf298 = buf290; del buf290  # reuse
        buf299 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf301 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf297, primals_203, primals_204, buf298, buf299, buf301, primals_203, primals_204, 768, 392, grid=grid(768), stream=stream0)
        del primals_203
        del primals_204
        buf302 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf303 = reinterpret_tensor(buf302, (8, 768), (768, 1), 0); del buf302  # reuse
        # Source Nodes: [x_169, x_170, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_mean_view_50.run(buf303, buf297, buf298, buf299, primals_118, primals_119, 6144, 49, grid=grid(6144), stream=stream0)
        del buf297
        del buf299
        del primals_119
        buf304 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_121, buf303, reinterpret_tensor(primals_120, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf304)
        del primals_121
        # Source Nodes: [l__mod___stem_1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_124, primals_124, 1, grid=grid(1), stream=stream0)
        del primals_124
        # Source Nodes: [x_3], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_127, primals_127, 1, grid=grid(1), stream=stream0)
        del primals_127
        # Source Nodes: [getattr_l__mod___stage1___0___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_130, primals_130, 1, grid=grid(1), stream=stream0)
        del primals_130
        # Source Nodes: [getattr_l__mod___stage1___1___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_133, primals_133, 1, grid=grid(1), stream=stream0)
        del primals_133
        # Source Nodes: [getattr_l__mod___stage1___2___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_136, primals_136, 1, grid=grid(1), stream=stream0)
        del primals_136
        # Source Nodes: [getattr_l__mod___stage1___3___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_139, primals_139, 1, grid=grid(1), stream=stream0)
        del primals_139
        # Source Nodes: [getattr_l__mod___stage1___4___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_142, primals_142, 1, grid=grid(1), stream=stream0)
        del primals_142
        # Source Nodes: [getattr_l__mod___stage1___5___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_145, primals_145, 1, grid=grid(1), stream=stream0)
        del primals_145
        # Source Nodes: [getattr_l__mod___stage1___6___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_148, primals_148, 1, grid=grid(1), stream=stream0)
        del primals_148
        # Source Nodes: [x_64], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_151, primals_151, 1, grid=grid(1), stream=stream0)
        del primals_151
        # Source Nodes: [getattr_l__mod___stage2___0___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_154, primals_154, 1, grid=grid(1), stream=stream0)
        del primals_154
        # Source Nodes: [getattr_l__mod___stage2___0___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_157, primals_157, 1, grid=grid(1), stream=stream0)
        del primals_157
        # Source Nodes: [getattr_l__mod___stage2___1___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_160, primals_160, 1, grid=grid(1), stream=stream0)
        del primals_160
        # Source Nodes: [getattr_l__mod___stage2___1___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_163, primals_163, 1, grid=grid(1), stream=stream0)
        del primals_163
        # Source Nodes: [getattr_l__mod___stage2___2___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_166, primals_166, 1, grid=grid(1), stream=stream0)
        del primals_166
        # Source Nodes: [getattr_l__mod___stage2___2___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_169, primals_169, 1, grid=grid(1), stream=stream0)
        del primals_169
        # Source Nodes: [getattr_l__mod___stage2___3___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_172, primals_172, 1, grid=grid(1), stream=stream0)
        del primals_172
        # Source Nodes: [getattr_l__mod___stage2___3___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_175, primals_175, 1, grid=grid(1), stream=stream0)
        del primals_175
        # Source Nodes: [x_117], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_178, primals_178, 1, grid=grid(1), stream=stream0)
        del primals_178
        # Source Nodes: [getattr_l__mod___stage3___0___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_181, primals_181, 1, grid=grid(1), stream=stream0)
        del primals_181
        # Source Nodes: [getattr_l__mod___stage3___0___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_184, primals_184, 1, grid=grid(1), stream=stream0)
        del primals_184
        # Source Nodes: [getattr_l__mod___stage3___1___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_187, primals_187, 1, grid=grid(1), stream=stream0)
        del primals_187
        # Source Nodes: [getattr_l__mod___stage3___1___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_190, primals_190, 1, grid=grid(1), stream=stream0)
        del primals_190
        # Source Nodes: [getattr_l__mod___stage3___2___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_193, primals_193, 1, grid=grid(1), stream=stream0)
        del primals_193
        # Source Nodes: [getattr_l__mod___stage3___2___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_196, primals_196, 1, grid=grid(1), stream=stream0)
        del primals_196
        # Source Nodes: [getattr_l__mod___stage3___3___norm1], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_199, primals_199, 1, grid=grid(1), stream=stream0)
        del primals_199
        # Source Nodes: [getattr_l__mod___stage3___3___norm2], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_202, primals_202, 1, grid=grid(1), stream=stream0)
        del primals_202
        # Source Nodes: [x_169], Original ATen: [aten.add]
        triton_poi_fused_add_51.run(primals_205, primals_205, 1, grid=grid(1), stream=stream0)
        del primals_205
        return (buf304, primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, buf0, buf7, buf8, buf10, buf14, buf15, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf80, buf81, buf82, buf83, buf84, buf85, buf87, buf89, buf93, buf94, buf98, buf99, buf109, buf110, buf114, buf115, buf116, buf117, buf118, buf122, buf123, buf133, buf134, buf138, buf139, buf140, buf141, buf142, buf147, buf148, buf158, buf159, buf163, buf164, buf165, buf166, buf167, buf171, buf172, buf182, buf183, buf187, buf188, buf189, buf190, buf192, buf194, buf198, buf199, buf203, buf204, buf214, buf215, buf219, buf220, buf221, buf222, buf223, buf227, buf228, buf238, buf239, buf243, buf244, buf245, buf246, buf247, buf252, buf253, buf263, buf264, buf268, buf269, buf270, buf271, buf272, buf276, buf277, buf287, buf288, buf292, buf293, buf294, buf295, buf296, buf301, buf303, reinterpret_tensor(primals_120, (1000, 768), (768, 1), 0), reinterpret_tensor(buf298, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf289, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf284, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf285, (48, 128, 49), (6272, 1, 128), 0), buf305, reinterpret_tensor(buf279, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf280, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf273, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf265, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf260, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf261, (48, 128, 49), (6272, 1, 128), 0), buf306, reinterpret_tensor(buf255, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf256, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf249, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf240, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf235, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf236, (48, 128, 49), (6272, 1, 128), 0), buf307, reinterpret_tensor(buf230, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf231, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf224, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf216, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf211, (48, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf212, (48, 128, 49), (6272, 1, 128), 0), buf308, reinterpret_tensor(buf206, (48, 128, 49), (6272, 1, 128), 0), reinterpret_tensor(buf207, (48, 49, 128), (6272, 1, 49), 0), reinterpret_tensor(buf200, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf195, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf184, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf179, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf180, (48, 64, 196), (12544, 1, 64), 0), buf309, reinterpret_tensor(buf174, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf175, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf168, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf160, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf155, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf156, (48, 64, 196), (12544, 1, 64), 0), buf310, reinterpret_tensor(buf150, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf151, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf144, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf135, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf130, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf131, (48, 64, 196), (12544, 1, 64), 0), buf311, reinterpret_tensor(buf125, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf126, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf119, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf106, (48, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf107, (48, 64, 196), (12544, 1, 64), 0), buf312, reinterpret_tensor(buf101, (48, 64, 196), (12544, 1, 64), 0), reinterpret_tensor(buf102, (48, 196, 64), (12544, 1, 196), 0), reinterpret_tensor(buf95, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf90, (1, 384, 1, 1), (384, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf67, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf46, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf26, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf16, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf11, (1, 192, 1, 1), (192, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_125 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_128 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_131 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_137 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_140 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_143 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_182 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_188 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
