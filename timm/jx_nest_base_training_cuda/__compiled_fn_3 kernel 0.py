
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


# kernel path: /tmp/torchinductor_youkaichao/ai/caiweu2df33ivpkdxbgujpighh73sdergpfja43lvvqcckzg5pon.py
# Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv, x_8, y], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv => view_2
# x_8 => add
# y => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tl.store(out_ptr2 + (r3 + (128*x5)), tmp20, rmask & xmask)
        tl.store(out_ptr3 + (r3 + (128*x5)), tmp24, rmask & xmask)
    tmp25 = 128.0
    tmp26 = tmp7 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = tl.math.rsqrt(tmp28)
    tmp30 = tmp29 / tmp25
    tl.store(out_ptr4 + (x5), tmp30, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/3b/c3b3upuiqojomfbacf7beuefakrysxnpnjvoajgylurabajpnunh.py
# Source Nodes: [x_12], Original ATen: [aten.view]
# x_12 => view_12
triton_poi_fused_view_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 3136)) + (100352*(x0 % 4)) + (401408*(x1 // 3136)) + (x0 // 4)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbbjrrrijdvednpdhvocghgo6y67iozo425scnyqk73ut52ikw2.py
# Source Nodes: [x_14, x_15, x_16, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# x_14 => add_3
# x_15 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
# x_16 => view_14
# x_8 => add
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r3 + (128*x5)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (128*x5)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (128*x5)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x5), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdrctmegqv7ut5kwzsyedoit2rovrtyhdxqegnl24ikymwnjcdem.py
# Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
# x_17 => add_6, erf, mul_6, mul_7, mul_8
# x_20 => view_16
triton_poi_fused_gelu_view_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vc56pg55mufasixktgkj63ryzsviuwjw4yvbsinkr5wiqquyh6.py
# Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv, x_22, y_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv => view_18
# x_22 => add_7
# y_1 => add_8, add_9, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5eq2at6ua7cj72wujnl3c4gdrz3rf6g5bccaifxsdi457afi2li.py
# Source Nodes: [random_tensor], Original ATen: [aten.bernoulli]
# random_tensor => bernoulli
triton_poi_fused_bernoulli_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bernoulli_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = float("nan")
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pp/cppq6dw4pbxsq5qt3td3z2s2czdbbkriqe7phstbhxolazu3xetf.py
# Source Nodes: [div_, mul, x_22, x_28, x_29, x_30], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div_ => div_2
# mul => mul_13
# x_22 => add_7
# x_28 => add_10
# x_29 => add_11, add_12, mul_14, mul_15, rsqrt_3, sub_5, var_mean_3
# x_30 => view_30
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = 0.9782608691602945
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp4 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp34 / tmp30
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp12, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp40, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywkdfjqne6gd42kcsigynyd2btlpymqjfbgs7u22m36p3q3outy.py
# Source Nodes: [permute_5], Original ATen: [aten.permute]
# permute_5 => permute_17
triton_poi_fused_permute_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_permute_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9782608691602945
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vi/cvioayg2c4ezbtnrofj4ta2gj26aihldmkjsj2tizsbjcs6w3lql.py
# Source Nodes: [x_41], Original ATen: [aten.convolution]
# x_41 => convolution_1
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6qm6zvhdadr56ivepttqtlzvqoendfcrjhgkbberausb2yypug.py
# Source Nodes: [x_42], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_42 => add_15, clone_16, mul_20, rsqrt_4, sub_6, var_mean_4
triton_red_fused_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp7 = tl.load(in_ptr0 + (x0 + (3136*r2) + (802816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 256.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-06
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp16, rmask & xmask)
    tmp17 = 256.0
    tmp18 = tmp5 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp21 / tmp17
    tl.store(out_ptr3 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqy2mheayl5akkj4hidguwdvv74mjfcivzqvvryakripi4icxque.py
# Source Nodes: [x_45], Original ATen: [aten.constant_pad_nd]
# x_45 => constant_pad_nd
triton_poi_fused_constant_pad_nd_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6653952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 14592) % 57
    x1 = (xindex // 256) % 57
    x3 = (xindex // 831744)
    x4 = xindex % 14592
    x0 = xindex % 256
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (802816*x3)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, float("-inf"), tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7ruroxaejbrgaz5qhdm3dcaubrn7mr62i6j3ymsbjmbt35szmh.py
# Source Nodes: [x_47], Original ATen: [aten.max_pool2d_with_indices]
# x_47 => getitem_17, max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 28
    x3 = (xindex // 28)
    y0 = yindex % 256
    y1 = (yindex // 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (256 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (512 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14592 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14848 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15104 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (29184 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (29440 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (29696 + y0 + (512*x2) + (29184*x3) + (831744*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x2) + (114*x3)
    tmp19 = (2*x2) + (114*x3)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x2) + (114*x3)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 57 + (2*x2) + (114*x3)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 58 + (2*x2) + (114*x3)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 59 + (2*x2) + (114*x3)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 114 + (2*x2) + (114*x3)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 115 + (2*x2) + (114*x3)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 116 + (2*x2) + (114*x3)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (y0 + (256*x4) + (200704*y1)), tmp16, xmask)
    tl.store(out_ptr1 + (y0 + (256*x4) + (200704*y1)), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxdtazmgoheduyzhgaxtanwozygc2we653e6og6nwj4e4ezqr2r.py
# Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv, x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv => view_38
# x_52 => add_17
# y_2 => add_18, add_19, mul_22, mul_23, rsqrt_5, sub_7, var_mean_5
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r3 + (256*(x0 % 14)) + (3584*(x1 % 2)) + (7168*(x0 // 14)) + (100352*(x1 // 2)) + (200704*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 256.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r3 + (256*x5)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (256*x5)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x5), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cquyeaqtd7zm7rhbll63qtts3i2myvk737yuwvtb4c47xj76zdhd.py
# Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
# x_54 => clone_18, mul_24
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wxfginwyjbhh5lwldr7hebmsqfst6dpcq35lyanlwfdetcdzee.py
# Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
# x_54 => clone_19, mul_25
triton_poi_fused_clone_mul_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_18', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bb7uhflvpnl3gmxv4swz3b7lqlkqsqa3jrzcnhdrfczw73qhyx.py
# Source Nodes: [x_54], Original ATen: [aten._softmax]
# x_54 => amax_2, div_4, exp_2, sub_8, sum_3
triton_per_fused__softmax_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxxwnuc2aatxx32vvj76maei2jt6bpa3wnjuxyxvg5shuo3pwlz.py
# Source Nodes: [x_54], Original ATen: [aten.clone]
# x_54 => clone_20
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
    x1 = (xindex // 32) % 784
    x2 = (xindex // 25088) % 8
    x3 = (xindex // 200704)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (32*x2) + (768*x1) + (602112*x3)), None)
    tmp1 = tl.load(in_ptr1 + (512 + x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkfmtbkvarxxrhw25vfunlvd24nbvvmzbtw4xldks4ql3at2xpv.py
# Source Nodes: [x_56], Original ATen: [aten.view]
# x_56 => view_48
triton_poi_fused_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 784)) + (25088*(x0 % 8)) + (200704*(x1 // 784)) + (x0 // 8)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4c4wubuluxrqxtv2wsc2zgiz66zyz5awtvvtia7vtzev2a5j5ia.py
# Source Nodes: [div__2, mul_2, x_52, x_58, x_59, x_60], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__2 => div_5
# mul_2 => mul_26
# x_52 => add_17
# x_58 => add_20
# x_59 => add_21, add_22, mul_27, mul_28, rsqrt_6, sub_9, var_mean_6
# x_60 => view_50
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_22', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r3 + (256*(x0 % 14)) + (3584*(x1 % 2)) + (7168*(x0 // 14)) + (100352*(x1 // 2)) + (200704*x2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r3 + (256*x5)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = 0.9565217383205891
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp2 + tmp9
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
    tmp38 = tmp32 / tmp28
    tl.store(in_out_ptr0 + (r3 + (256*x5)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (256*x5)), tmp33, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (256*x5)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (x5), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhrjp3bbpceijpqlaieg7qinogooh2o4ziyolg2milub7mpoplc.py
# Source Nodes: [x_61, x_64], Original ATen: [aten.gelu, aten.view]
# x_61 => add_23, erf_2, mul_29, mul_30, mul_31
# x_64 => view_52
triton_poi_fused_gelu_view_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/hq/chqupg3p6pwq26ppjzbjkuvr2kfpgnqhl5zfrdpld2qrojo3c23t.py
# Source Nodes: [div__3, getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv, mul_3, x_66, y_3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__3 => div_6
# getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv => view_54
# mul_3 => mul_32
# x_66 => add_24
# y_3 => add_25, add_26, mul_33, mul_34, rsqrt_7, sub_10, var_mean_7
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9565217383205891
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp25 = tmp8 - tmp18
    tmp26 = 256.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5f6u4fpf6okgx64fonrcahhksltwgxew3nbooskamecx77wecr.py
# Source Nodes: [div__3, div__4, mul_3, mul_4, x_66, x_72, x_73, x_74], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__3 => div_6
# div__4 => div_8
# mul_3 => mul_32
# mul_4 => mul_37
# x_66 => add_24
# x_72 => add_27
# x_73 => add_28, add_29, mul_38, mul_39, rsqrt_8, sub_12, var_mean_8
# x_74 => view_66
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9565217383205891
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.9347826093435287
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 256, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 256.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnk5u3qnmuksvmv2vdczwbx3kdqafgoctuuf4gayw7ywyfnz6n7x.py
# Source Nodes: [permute_13], Original ATen: [aten.permute]
# permute_13 => permute_37
triton_poi_fused_permute_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_permute_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9347826093435287
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/csspndw4ek2u5oprz5nexuqwcjqvhzwrtix6lgzcsouf2uh4w7vp.py
# Source Nodes: [x_85], Original ATen: [aten.convolution]
# x_85 => convolution_2
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3ex5o4cwxhwuo452pbaeu4i3m7bf5xekgq7uhnkpe254frkcol.py
# Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
# x_86 => clone_33, var_mean_9
triton_red_fused_native_layer_norm_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_28', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xt/cxteywpygmxnqoxzflg5qujsmymncyu426e56sjjpakksodw7mdc.py
# Source Nodes: [x_86], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_86 => add_32, clone_33, rsqrt_9, var_mean_9
triton_per_fused_native_layer_norm_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x3), tmp21, xmask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhlxbajcceqt4lfd5sut4jjacgxn6lmcokgadph72wmo5xpdeoe.py
# Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
# x_86 => add_32, clone_33, mul_44, rsqrt_9, sub_13, var_mean_9
triton_poi_fused_native_layer_norm_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 512.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxw7nelvokoyackxkfb6yw26ay5nmnb2ztufklwrvmftodu22hrg.py
# Source Nodes: [x_89], Original ATen: [aten.constant_pad_nd]
# x_89 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3444736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 14848) % 29
    x1 = (xindex // 512) % 29
    x3 = (xindex // 430592)
    x4 = xindex % 14848
    x0 = xindex % 512
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (401408*x3)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, float("-inf"), tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxk77x45sfpptz6nlnngipub2dlolidxe7bb3kbksiu2e3j3rvl5.py
# Source Nodes: [x_91], Original ATen: [aten.max_pool2d_with_indices]
# x_91 => getitem_35, max_pool2d_with_indices_1
triton_poi_fused_max_pool2d_with_indices_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y0 = yindex % 512
    y1 = (yindex // 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (512 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1024 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14848 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (15360 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15872 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (29696 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (30208 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (30720 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x2) + (58*x3)
    tmp19 = (2*x2) + (58*x3)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x2) + (58*x3)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 29 + (2*x2) + (58*x3)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 30 + (2*x2) + (58*x3)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 31 + (2*x2) + (58*x3)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 58 + (2*x2) + (58*x3)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 59 + (2*x2) + (58*x3)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 60 + (2*x2) + (58*x3)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (y0 + (512*x4) + (100352*y1)), tmp16, xmask)
    tl.store(out_ptr1 + (y0 + (512*x4) + (100352*y1)), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xt/cxt2u5azntwdxr5sha3fvhwg6bla6j54jeeqek6pkl5syol34gyj.py
# Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv, x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv => view_74
# x_96 => add_34
# y_4 => add_35, add_36, mul_46, mul_47, rsqrt_10, sub_14, var_mean_10
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp2 - tmp12
    tmp20 = 512.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = tl.math.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 / tmp20
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp29, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7c/c7cxllzb4n4qdyncf5afzrkifdzqidvsg4g5bmmxikvs6pzhztfq.py
# Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
# x_98 => clone_34, mul_48
triton_poi_fused_clone_mul_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_34', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kw/ckw73fpjearbxy4u7bgrkcbyoylfjaaenq4mm4b5uma56mggqplz.py
# Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
# x_98 => clone_35, mul_49
triton_poi_fused_clone_mul_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_35', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqdwff27qykpwpip6rbook2zoo3uaspnyzlowiwuuuy7ftfpfoo.py
# Source Nodes: [x_98], Original ATen: [aten._softmax]
# x_98 => amax_4, div_10, exp_4, sub_15, sum_5
triton_per_fused__softmax_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_36', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvx7p5rzz52eulmlgxntujyiszeu3en7cjrwy3snmvgh4plder5.py
# Source Nodes: [x_98], Original ATen: [aten.clone]
# x_98 => clone_36
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rkdwz57jyh4b27ck7sk7cdlmk3zn6thxcvt6eokskcqeimql6b.py
# Source Nodes: [x_100], Original ATen: [aten.view]
# x_100 => view_84
triton_poi_fused_view_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 196)) + (6272*(x0 % 16)) + (100352*(x1 // 196)) + (x0 // 16)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czrymrspstpmzr3b7qigcobtne55gwkydzx3nec4qt6tmv6gnzm6.py
# Source Nodes: [div__6, mul_6, x_102, x_103, x_104, x_96], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__6 => div_11
# mul_6 => mul_50
# x_102 => add_37
# x_103 => add_38, add_39, mul_51, mul_52, rsqrt_11, sub_16, var_mean_11
# x_104 => view_86
# x_96 => add_34
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = 0.9130434766411781
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tmp2 + tmp9
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
    tmp38 = tmp32 / tmp28
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp33, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp37, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnndtegd6eamm4rq2qiylws6pwme2ufsaubaa4yeirikt6h7fdng.py
# Source Nodes: [x_105, x_108], Original ATen: [aten.gelu, aten.view]
# x_105 => add_40, erf_4, mul_53, mul_54, mul_55
# x_108 => view_88
triton_poi_fused_gelu_view_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csfxbnbciwm24wkdsckvrmxd65i3vtjgpa3j4gy3wtolzvzcxei6.py
# Source Nodes: [div__7, getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv, mul_7, x_110, y_5], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__7 => div_12
# getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv => view_90
# mul_7 => mul_56
# x_110 => add_41
# y_5 => add_42, add_43, mul_57, mul_58, rsqrt_12, sub_17, var_mean_12
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9130434766411781
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coan5ei6mdbxtwpmnlmhwpv2qfgwgxt7eexcndx2plbpxk3eusfy.py
# Source Nodes: [div__7, div__8, mul_7, mul_8, x_110, x_116, x_117, x_118], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__7 => div_12
# div__8 => div_14
# mul_7 => mul_56
# mul_8 => mul_61
# x_110 => add_41
# x_116 => add_44
# x_117 => add_45, add_46, mul_62, mul_63, rsqrt_13, sub_19, var_mean_13
# x_118 => view_102
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_42', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9130434766411781
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.8913043439388275
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/ccimvdobazkiz3ufgjfpmiaaqrn545nwgdmghraanb7ovffyldf6.py
# Source Nodes: [div__9, getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv, mul_9, x_124, y_6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__9 => div_15
# getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv => view_106
# mul_9 => mul_67
# x_124 => add_48
# y_6 => add_49, add_50, mul_68, mul_69, rsqrt_14, sub_20, var_mean_14
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8913043439388275
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbzzapbqmmwhxjurqgpmryrz2scw5hlfkvq2gkwm7m5eja35kwa.py
# Source Nodes: [div__10, div__9, mul_10, mul_9, x_124, x_130, x_131, x_132], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__10 => div_17
# div__9 => div_15
# mul_10 => mul_72
# mul_9 => mul_67
# x_124 => add_48
# x_130 => add_51
# x_131 => add_52, add_53, mul_73, mul_74, rsqrt_15, sub_22, var_mean_15
# x_132 => view_118
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8913043439388275
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.8695652186870575
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl677o4fuusqgxmmdlvvglhqpwwgm5bn44vw2oczqhin4ltoitdb.py
# Source Nodes: [div__11, getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv, mul_11, x_138, y_7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__11 => div_18
# getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv => view_122
# mul_11 => mul_78
# x_138 => add_55
# y_7 => add_56, add_57, mul_79, mul_80, rsqrt_16, sub_23, var_mean_16
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8695652186870575
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvte72egjmznvuksddrp5id56lxduq6umhpywrdsgmv6cfiaftpn.py
# Source Nodes: [div__11, div__12, mul_11, mul_12, x_138, x_144, x_145, x_146], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__11 => div_18
# div__12 => div_20
# mul_11 => mul_78
# mul_12 => mul_83
# x_138 => add_55
# x_144 => add_58
# x_145 => add_59, add_60, mul_84, mul_85, rsqrt_17, sub_25, var_mean_17
# x_146 => view_134
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8695652186870575
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.8478260785341263
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ywbsvtqk23ql43h72w4xegb43kgxzoff2c73bq6ulr5gnlx4ks.py
# Source Nodes: [div__13, getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv, mul_13, x_152, y_8], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__13 => div_21
# getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv => view_138
# mul_13 => mul_89
# x_152 => add_62
# y_8 => add_63, add_64, mul_90, mul_91, rsqrt_18, sub_26, var_mean_18
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8478260785341263
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4osmublfgbb3dw7rckturbbo3gqylfxzesmcw4oebvd6xetm4g.py
# Source Nodes: [div__13, div__14, mul_13, mul_14, x_152, x_158, x_159, x_160], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__13 => div_21
# div__14 => div_23
# mul_13 => mul_89
# mul_14 => mul_94
# x_152 => add_62
# x_158 => add_65
# x_159 => add_66, add_67, mul_95, mul_96, rsqrt_19, sub_28, var_mean_19
# x_160 => view_150
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_48', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8478260785341263
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.8260869532823563
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotttkue2u72er3rkcdi5josrirotgl6pdovxgmsnroml43ogao7.py
# Source Nodes: [div__15, getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv, mul_15, x_166, y_9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__15 => div_24
# getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv => view_154
# mul_15 => mul_100
# x_166 => add_69
# y_9 => add_70, add_71, mul_101, mul_102, rsqrt_20, sub_29, var_mean_20
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8260869532823563
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ra/cralwqsrm7ptj3dzf52o5ixpaxwtqpwovdecmz7q7qm64ohlryfe.py
# Source Nodes: [div__15, div__16, mul_15, mul_16, x_166, x_172, x_173, x_174], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__15 => div_24
# div__16 => div_26
# mul_15 => mul_100
# mul_16 => mul_105
# x_166 => add_69
# x_172 => add_72
# x_173 => add_73, add_74, mul_106, mul_107, rsqrt_21, sub_31, var_mean_21
# x_174 => view_166
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8260869532823563
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.8043478280305862
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfujxwtqg52p4jdynvefqydgkgffmpy5zlqxkrizghl6qcwoeb2.py
# Source Nodes: [div__17, getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv, mul_17, x_180, y_10], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__17 => div_27
# getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv => view_170
# mul_17 => mul_111
# x_180 => add_76
# y_10 => add_77, add_78, mul_112, mul_113, rsqrt_22, sub_32, var_mean_22
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8043478280305862
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ces27zmfbc3f56sy36qrmpusdkfeq22nsidrelbcbivdfo252ftx.py
# Source Nodes: [div__17, div__18, mul_17, mul_18, x_180, x_186, x_187, x_188], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__17 => div_27
# div__18 => div_29
# mul_17 => mul_111
# mul_18 => mul_116
# x_180 => add_76
# x_186 => add_79
# x_187 => add_80, add_81, mul_117, mul_118, rsqrt_23, sub_34, var_mean_23
# x_188 => view_182
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.8043478280305862
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.782608687877655
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdygfzieohlpbrylgjzcr7a3g6oib7h3hnmz54tm5pq6llvu3h6z.py
# Source Nodes: [div__19, getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv, mul_19, x_194, y_11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__19 => div_30
# getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv => view_186
# mul_19 => mul_122
# x_194 => add_83
# y_11 => add_84, add_85, mul_123, mul_124, rsqrt_24, sub_35, var_mean_24
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.782608687877655
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/cten7wq5nz5dvx6ajybrlaberxykjo4holzm6isic5huavvv3iwa.py
# Source Nodes: [div__19, div__20, mul_19, mul_20, x_194, x_200, x_201, x_202], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__19 => div_30
# div__20 => div_32
# mul_19 => mul_122
# mul_20 => mul_127
# x_194 => add_83
# x_200 => add_86
# x_201 => add_87, add_88, mul_128, mul_129, rsqrt_25, sub_37, var_mean_25
# x_202 => view_198
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.782608687877655
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.760869562625885
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4coma43jqbozgszd26awb7aug3dcviciwucwih4o7qt6suvzdjt.py
# Source Nodes: [div__21, getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv, mul_21, x_208, y_12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__21 => div_33
# getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv => view_202
# mul_21 => mul_133
# x_208 => add_90
# y_12 => add_91, add_92, mul_134, mul_135, rsqrt_26, sub_38, var_mean_26
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.760869562625885
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf35kiichclaggl4myixe2e53mcouzy7bf346bb62dnvwt735wqp.py
# Source Nodes: [div__21, div__22, mul_21, mul_22, x_208, x_214, x_215, x_216], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__21 => div_33
# div__22 => div_35
# mul_21 => mul_133
# mul_22 => mul_138
# x_208 => add_90
# x_214 => add_93
# x_215 => add_94, add_95, mul_139, mul_140, rsqrt_27, sub_40, var_mean_27
# x_216 => view_214
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.760869562625885
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.739130437374115
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgti7blle4buweq4bxygovhifsa52bf3o4vil4ghq5yu6m5dk5c.py
# Source Nodes: [div__23, getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv, mul_23, x_222, y_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__23 => div_36
# getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv => view_218
# mul_23 => mul_144
# x_222 => add_97
# y_13 => add_98, add_99, mul_145, mul_146, rsqrt_28, sub_41, var_mean_28
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.739130437374115
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55dpky4bgmiwnbovjdkeu2kococxboig2mh5alkybc22v3bywka.py
# Source Nodes: [div__23, div__24, mul_23, mul_24, x_222, x_228, x_229, x_230], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__23 => div_36
# div__24 => div_38
# mul_23 => mul_144
# mul_24 => mul_149
# x_222 => add_97
# x_228 => add_100
# x_229 => add_101, add_102, mul_150, mul_151, rsqrt_29, sub_43, var_mean_29
# x_230 => view_230
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.739130437374115
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.717391312122345
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32oc4fuujk3rmw4sy25ji5y3luggzpm3irkrnpi2ecyzzepqkku.py
# Source Nodes: [div__25, getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv, mul_25, x_236, y_14], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__25 => div_39
# getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv => view_234
# mul_25 => mul_155
# x_236 => add_104
# y_14 => add_105, add_106, mul_156, mul_157, rsqrt_30, sub_44, var_mean_30
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.717391312122345
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dmo6nvwf3ytnyp77cekyvr6afaw5b3q56lx3c3la36r7kvawyy.py
# Source Nodes: [div__25, div__26, mul_25, mul_26, x_236, x_242, x_243, x_244], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__25 => div_39
# div__26 => div_41
# mul_25 => mul_155
# mul_26 => mul_160
# x_236 => add_104
# x_242 => add_107
# x_243 => add_108, add_109, mul_161, mul_162, rsqrt_31, sub_46, var_mean_31
# x_244 => view_246
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_60', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.717391312122345
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.695652186870575
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lta3xnarl3fd4lhizd6ouht5xcrpyyjhno25oh6dvodql3gocc.py
# Source Nodes: [div__27, getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv, mul_27, x_250, y_15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__27 => div_42
# getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv => view_250
# mul_27 => mul_166
# x_250 => add_111
# y_15 => add_112, add_113, mul_167, mul_168, rsqrt_32, sub_47, var_mean_32
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.695652186870575
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhqltax2cbzzgxrnhfdadx6khykn2rwdwdkb6waz7sg2yms2cok.py
# Source Nodes: [div__27, div__28, mul_27, mul_28, x_250, x_256, x_257, x_258], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__27 => div_42
# div__28 => div_44
# mul_27 => mul_166
# mul_28 => mul_171
# x_250 => add_111
# x_256 => add_114
# x_257 => add_115, add_116, mul_172, mul_173, rsqrt_33, sub_49, var_mean_33
# x_258 => view_262
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.695652186870575
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.6739130616188049
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/ctiwgklr7shxynuiwzgb2cneifm6dyh462ekg3w72avi6fsfdjag.py
# Source Nodes: [div__29, getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv, mul_29, x_264, y_16], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__29 => div_45
# getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv => view_266
# mul_29 => mul_177
# x_264 => add_118
# y_16 => add_119, add_120, mul_178, mul_179, rsqrt_34, sub_50, var_mean_34
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6739130616188049
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7ynoepv4vfsvfgm56tbbzjquiuy6itelo4kabgfyqjspldkury.py
# Source Nodes: [div__29, div__30, mul_29, mul_30, x_264, x_270, x_271, x_272], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__29 => div_45
# div__30 => div_47
# mul_29 => mul_177
# mul_30 => mul_182
# x_264 => add_118
# x_270 => add_121
# x_271 => add_122, add_123, mul_183, mul_184, rsqrt_35, sub_52, var_mean_35
# x_272 => view_278
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6739130616188049
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.6521739065647125
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hh/chhpiohqiu7shswv7arh4qcjos7lpjz7a5pvkpccz4l4ireh7diq.py
# Source Nodes: [div__31, getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv, mul_31, x_278, y_17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__31 => div_48
# getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv => view_282
# mul_31 => mul_188
# x_278 => add_125
# y_17 => add_126, add_127, mul_189, mul_190, rsqrt_36, sub_53, var_mean_36
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6521739065647125
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdr6wvnatz6m2otyvuwdf3gygciw2ucksjvcdcha5m3ey7zijb7.py
# Source Nodes: [div__31, div__32, mul_31, mul_32, x_278, x_284, x_285, x_286], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__31 => div_48
# div__32 => div_50
# mul_31 => mul_188
# mul_32 => mul_193
# x_278 => add_125
# x_284 => add_128
# x_285 => add_129, add_130, mul_194, mul_195, rsqrt_37, sub_55, var_mean_37
# x_286 => view_294
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_66', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6521739065647125
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.6304347813129425
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csb34tvzrk72bu6tzzls7tavbiibhu3kldxspjbl7sezit5ldduv.py
# Source Nodes: [div__33, getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv, mul_33, x_292, y_18], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__33 => div_51
# getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv => view_298
# mul_33 => mul_199
# x_292 => add_132
# y_18 => add_133, add_134, mul_200, mul_201, rsqrt_38, sub_56, var_mean_38
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6304347813129425
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpusss3njj7uj4f6ymwwk2w3w4z6vp62bn5uvwsmr2ceqsly4pl.py
# Source Nodes: [div__33, div__34, mul_33, mul_34, x_292, x_298, x_299, x_300], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__33 => div_51
# div__34 => div_53
# mul_33 => mul_199
# mul_34 => mul_204
# x_292 => add_132
# x_298 => add_135
# x_299 => add_136, add_137, mul_205, mul_206, rsqrt_39, sub_58, var_mean_39
# x_300 => view_310
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_68', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6304347813129425
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.6086956560611725
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curszkuvmosrv5pbxgggau7s55mzk36iubhanjtoovir7hftn6sc.py
# Source Nodes: [div__35, getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv, mul_35, x_306, y_19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__35 => div_54
# getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv => view_314
# mul_35 => mul_210
# x_306 => add_139
# y_19 => add_140, add_141, mul_211, mul_212, rsqrt_40, sub_59, var_mean_40
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6086956560611725
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc2g5lurokm3whrddjimjmhactoq5ecxh2euwy7k4oxytckbr7mm.py
# Source Nodes: [div__35, div__36, mul_35, mul_36, x_306, x_312, x_313, x_314], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__35 => div_54
# div__36 => div_56
# mul_35 => mul_210
# mul_36 => mul_215
# x_306 => add_139
# x_312 => add_142
# x_313 => add_143, add_144, mul_216, mul_217, rsqrt_41, sub_61, var_mean_41
# x_314 => view_326
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.6086956560611725
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.5869565308094025
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbpjad4yqy6sxnruaid3jqb3g4zrncch7tfgp6y6xjyfoutxga6.py
# Source Nodes: [div__37, getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv, mul_37, x_320, y_20], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__37 => div_57
# getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv => view_330
# mul_37 => mul_221
# x_320 => add_146
# y_20 => add_147, add_148, mul_222, mul_223, rsqrt_42, sub_62, var_mean_42
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.5869565308094025
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckehts7buucxmpalqejse5ew5qvt7vytj22j2ljfjlq22v2b5gu7.py
# Source Nodes: [div__37, div__38, mul_37, mul_38, x_320, x_326, x_327, x_328], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__37 => div_57
# div__38 => div_59
# mul_37 => mul_221
# mul_38 => mul_226
# x_320 => add_146
# x_326 => add_149
# x_327 => add_150, add_151, mul_227, mul_228, rsqrt_43, sub_64, var_mean_43
# x_328 => view_342
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.5869565308094025
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.5652174055576324
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/ccccbnadvp2355kxwnvcpdkcmbdpe7miu6w3ttjalgylua45bquv.py
# Source Nodes: [div__39, getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv, mul_39, x_334, y_21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__39 => div_60
# getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv => view_346
# mul_39 => mul_232
# x_334 => add_153
# y_21 => add_154, add_155, mul_233, mul_234, rsqrt_44, sub_65, var_mean_44
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.5652174055576324
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/crop46o4rcdt3bzs3cm5ldgr5tdqox5kj4rvwopect26cwsgorbc.py
# Source Nodes: [div__39, div__40, mul_39, mul_40, x_334, x_340, x_341, x_342], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__39 => div_60
# div__40 => div_62
# mul_39 => mul_232
# mul_40 => mul_237
# x_334 => add_153
# x_340 => add_156
# x_341 => add_157, add_158, mul_238, mul_239, rsqrt_45, sub_67, var_mean_45
# x_342 => view_358
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_74', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.5652174055576324
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.54347825050354
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dqsnsxrb6owjgqprrr33gutu5xppu4zy6fxpoihhnvu6pengiy.py
# Source Nodes: [div__41, getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv, mul_41, x_348, y_22], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__41 => div_63
# getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv => view_362
# mul_41 => mul_243
# x_348 => add_160
# y_22 => add_161, add_162, mul_244, mul_245, rsqrt_46, sub_68, var_mean_46
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.54347825050354
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwrciminprp7ocu22tje2rr33afennfernq3m4k52u22bd4l3tr.py
# Source Nodes: [div__41, div__42, mul_41, mul_42, x_348, x_354, x_355, x_356], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__41 => div_63
# div__42 => div_65
# mul_41 => mul_243
# mul_42 => mul_248
# x_348 => add_160
# x_354 => add_163
# x_355 => add_164, add_165, mul_249, mul_250, rsqrt_47, sub_70, var_mean_47
# x_356 => view_374
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_76', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.54347825050354
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.52173912525177
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewvukvxg6uwh4diraapmmqs52hqd5g75ntqbtceosbsqsfbhqea.py
# Source Nodes: [div__43, getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv, mul_43, x_362, y_23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__43 => div_66
# getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv => view_378
# mul_43 => mul_254
# x_362 => add_167
# y_23 => add_168, add_169, mul_255, mul_256, rsqrt_48, sub_71, var_mean_48
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.52173912525177
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp36 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkm4ba4fe5itnxxc7fhtbsb2y64axwbkzofgehthxg3a2zo5wm4.py
# Source Nodes: [div__43, div__44, mul_43, mul_44, x_362, x_368, x_369, x_370], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# div__43 => div_66
# div__44 => div_68
# mul_43 => mul_254
# mul_44 => mul_259
# x_362 => add_167
# x_368 => add_170
# x_369 => add_171, add_172, mul_260, mul_261, rsqrt_49, sub_73, var_mean_49
# x_370 => view_390
triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_78', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = 0.52173912525177
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = 0.5
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 512, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 512.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-06
    tmp37 = tmp35 + tmp36
    tmp38 = tl.math.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp38 / tmp34
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp16, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (512*x3)), tmp43, rmask & xmask)
    tl.store(out_ptr4 + (x3), tmp44, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3z7ybexinsvhst7oxxgyigokfjrfxsdiyfbqa76ge6p5q4spnz.py
# Source Nodes: [x_382], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_382 => add_175, mul_266, rsqrt_50, sub_74, var_mean_50
triton_per_fused_native_layer_norm_native_layer_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
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
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.5
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
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
    tmp32 = tmp30 / tmp26
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpor5ha3ipngxs5byyriw7huh2zzwwsjhaghie4gy6p5thukr75g.py
# Source Nodes: [x_385], Original ATen: [aten.mean]
# x_385 => mean
triton_red_fused_mean_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7uz54lxduomwn37tzg5poviz2o6tt4aqy6iv4ji6cncv3oyg6w.py
# Source Nodes: [x_385], Original ATen: [aten.mean]
# x_385 => mean
triton_per_fused_mean_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_81', 'mutated_arg_names': ['in_out_ptr0']}
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (1, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (1, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (384, 128), (128, 1))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (128, 128), (128, 1))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (512, 128), (128, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (128, 512), (512, 1))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (384, 128), (128, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (128, 128), (128, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (512, 128), (128, 1))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (128, 512), (512, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (768, 256), (256, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (256, 256), (256, 1))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (1024, 256), (256, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (256, 1024), (1024, 1))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (768, 256), (256, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (256, 256), (256, 1))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (1024, 256), (256, 1))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (256, 1024), (1024, 1))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (1536, 512), (512, 1))
    assert_size_stride(primals_145, (1536, ), (1, ))
    assert_size_stride(primals_146, (512, 512), (512, 1))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512), (512, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (512, 2048), (2048, 1))
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (1536, 512), (512, 1))
    assert_size_stride(primals_153, (1536, ), (1, ))
    assert_size_stride(primals_154, (512, 512), (512, 1))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (2048, 512), (512, 1))
    assert_size_stride(primals_157, (2048, ), (1, ))
    assert_size_stride(primals_158, (512, 2048), (2048, 1))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (1536, 512), (512, 1))
    assert_size_stride(primals_161, (1536, ), (1, ))
    assert_size_stride(primals_162, (512, 512), (512, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (2048, 512), (512, 1))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_166, (512, 2048), (2048, 1))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (1536, 512), (512, 1))
    assert_size_stride(primals_169, (1536, ), (1, ))
    assert_size_stride(primals_170, (512, 512), (512, 1))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (2048, 512), (512, 1))
    assert_size_stride(primals_173, (2048, ), (1, ))
    assert_size_stride(primals_174, (512, 2048), (2048, 1))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (1536, 512), (512, 1))
    assert_size_stride(primals_177, (1536, ), (1, ))
    assert_size_stride(primals_178, (512, 512), (512, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (2048, 512), (512, 1))
    assert_size_stride(primals_181, (2048, ), (1, ))
    assert_size_stride(primals_182, (512, 2048), (2048, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (1536, 512), (512, 1))
    assert_size_stride(primals_185, (1536, ), (1, ))
    assert_size_stride(primals_186, (512, 512), (512, 1))
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (2048, 512), (512, 1))
    assert_size_stride(primals_189, (2048, ), (1, ))
    assert_size_stride(primals_190, (512, 2048), (2048, 1))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (1536, 512), (512, 1))
    assert_size_stride(primals_193, (1536, ), (1, ))
    assert_size_stride(primals_194, (512, 512), (512, 1))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (2048, 512), (512, 1))
    assert_size_stride(primals_197, (2048, ), (1, ))
    assert_size_stride(primals_198, (512, 2048), (2048, 1))
    assert_size_stride(primals_199, (512, ), (1, ))
    assert_size_stride(primals_200, (1536, 512), (512, 1))
    assert_size_stride(primals_201, (1536, ), (1, ))
    assert_size_stride(primals_202, (512, 512), (512, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (2048, 512), (512, 1))
    assert_size_stride(primals_205, (2048, ), (1, ))
    assert_size_stride(primals_206, (512, 2048), (2048, 1))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (1536, 512), (512, 1))
    assert_size_stride(primals_209, (1536, ), (1, ))
    assert_size_stride(primals_210, (512, 512), (512, 1))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (2048, 512), (512, 1))
    assert_size_stride(primals_213, (2048, ), (1, ))
    assert_size_stride(primals_214, (512, 2048), (2048, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (1536, 512), (512, 1))
    assert_size_stride(primals_217, (1536, ), (1, ))
    assert_size_stride(primals_218, (512, 512), (512, 1))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (2048, 512), (512, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_222, (512, 2048), (2048, 1))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (1536, 512), (512, 1))
    assert_size_stride(primals_225, (1536, ), (1, ))
    assert_size_stride(primals_226, (512, 512), (512, 1))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_228, (2048, 512), (512, 1))
    assert_size_stride(primals_229, (2048, ), (1, ))
    assert_size_stride(primals_230, (512, 2048), (2048, 1))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (1536, 512), (512, 1))
    assert_size_stride(primals_233, (1536, ), (1, ))
    assert_size_stride(primals_234, (512, 512), (512, 1))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (2048, 512), (512, 1))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_238, (512, 2048), (2048, 1))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (1536, 512), (512, 1))
    assert_size_stride(primals_241, (1536, ), (1, ))
    assert_size_stride(primals_242, (512, 512), (512, 1))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (2048, 512), (512, 1))
    assert_size_stride(primals_245, (2048, ), (1, ))
    assert_size_stride(primals_246, (512, 2048), (2048, 1))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_248, (1536, 512), (512, 1))
    assert_size_stride(primals_249, (1536, ), (1, ))
    assert_size_stride(primals_250, (512, 512), (512, 1))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_252, (2048, 512), (512, 1))
    assert_size_stride(primals_253, (2048, ), (1, ))
    assert_size_stride(primals_254, (512, 2048), (2048, 1))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (1536, 512), (512, 1))
    assert_size_stride(primals_257, (1536, ), (1, ))
    assert_size_stride(primals_258, (512, 512), (512, 1))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_260, (2048, 512), (512, 1))
    assert_size_stride(primals_261, (2048, ), (1, ))
    assert_size_stride(primals_262, (512, 2048), (2048, 1))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (1536, 512), (512, 1))
    assert_size_stride(primals_265, (1536, ), (1, ))
    assert_size_stride(primals_266, (512, 512), (512, 1))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (2048, 512), (512, 1))
    assert_size_stride(primals_269, (2048, ), (1, ))
    assert_size_stride(primals_270, (512, 2048), (2048, 1))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (1536, 512), (512, 1))
    assert_size_stride(primals_273, (1536, ), (1, ))
    assert_size_stride(primals_274, (512, 512), (512, 1))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (2048, 512), (512, 1))
    assert_size_stride(primals_277, (2048, ), (1, ))
    assert_size_stride(primals_278, (512, 2048), (2048, 1))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (1536, 512), (512, 1))
    assert_size_stride(primals_281, (1536, ), (1, ))
    assert_size_stride(primals_282, (512, 512), (512, 1))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (2048, 512), (512, 1))
    assert_size_stride(primals_285, (2048, ), (1, ))
    assert_size_stride(primals_286, (512, 2048), (2048, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (1536, 512), (512, 1))
    assert_size_stride(primals_289, (1536, ), (1, ))
    assert_size_stride(primals_290, (512, 512), (512, 1))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (2048, 512), (512, 1))
    assert_size_stride(primals_293, (2048, ), (1, ))
    assert_size_stride(primals_294, (512, 2048), (2048, 1))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (1536, 512), (512, 1))
    assert_size_stride(primals_297, (1536, ), (1, ))
    assert_size_stride(primals_298, (512, 512), (512, 1))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (2048, 512), (512, 1))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (512, 2048), (2048, 1))
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (1000, 512), (512, 1))
    assert_size_stride(primals_305, (1000, ), (1, ))
    assert_size_stride(primals_306, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_306, primals_106, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf4 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        buf5 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf821 = empty((8, 16, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv, x_8, y], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        stream0 = get_cuda_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_view_0.run(buf0, primals_107, primals_1, primals_2, primals_3, buf4, buf5, buf821, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_3
        buf6 = empty((25088, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf5, reinterpret_tensor(primals_108, (128, 384), (1, 128), 0), out=buf6)
        buf7 = empty((8, 4, 16, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_1.run(buf6, primals_109, buf7, 3211264, grid=grid(3211264), stream=stream0)
        buf8 = empty((8, 4, 16, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf6, primals_109, buf8, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf9 = empty((512, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf8, (512, 32, 196), (6272, 196, 1), 0), out=buf9)
        buf12 = empty((8, 4, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf9, buf12, 100352, 196, grid=grid(100352), stream=stream0)
        buf13 = empty((8, 4, 16, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf6, primals_109, buf13, 3211264, grid=grid(3211264), stream=stream0)
        del primals_109
        buf14 = empty((512, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf13, (512, 196, 32), (6272, 32, 1), 0), out=buf14)
        buf15 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf14, buf15, 3211264, grid=grid(3211264), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (25088, 128), (128, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf15, reinterpret_tensor(primals_110, (128, 128), (1, 128), 0), out=buf16)
        buf17 = reinterpret_tensor(buf16, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf16  # reuse
        buf21 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        buf22 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf820 = empty((8, 16, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14, x_15, x_16, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_6.run(buf17, buf0, primals_107, primals_1, primals_111, primals_4, primals_5, buf21, buf22, buf820, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_1
        del primals_107
        del primals_111
        del primals_5
        buf23 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_113, buf22, reinterpret_tensor(primals_112, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf23)
        del primals_113
        buf24 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf23, buf24, 12845056, grid=grid(12845056), stream=stream0)
        buf25 = reinterpret_tensor(buf0, (25088, 128), (128, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf24, reinterpret_tensor(primals_114, (512, 128), (1, 512), 0), out=buf25)
        buf29 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        buf30 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf819 = empty((8, 16, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv, x_22, y_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_8.run(buf17, buf25, primals_115, primals_6, primals_7, buf29, buf30, buf819, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_7
        buf31 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf30, reinterpret_tensor(primals_116, (128, 384), (1, 128), 0), out=buf31)
        buf32 = empty((8, 4, 16, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_1.run(buf31, primals_117, buf32, 3211264, grid=grid(3211264), stream=stream0)
        buf33 = empty((8, 4, 16, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_2.run(buf31, primals_117, buf33, 16384, 196, grid=grid(16384, 196), stream=stream0)
        buf34 = buf9; del buf9  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf33, (512, 32, 196), (6272, 196, 1), 0), out=buf34)
        buf37 = empty((8, 4, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf34, buf37, 100352, 196, grid=grid(100352), stream=stream0)
        del buf34
        buf38 = empty((8, 4, 16, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf31, primals_117, buf38, 3211264, grid=grid(3211264), stream=stream0)
        del buf31
        del primals_117
        buf39 = empty((512, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf38, (512, 196, 32), (6272, 32, 1), 0), out=buf39)
        buf40 = empty((25088, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten.view]
        triton_poi_fused_view_5.run(buf39, buf40, 3211264, grid=grid(3211264), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (25088, 128), (128, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_118, (128, 128), (1, 128), 0), out=buf41)
        buf43 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf43, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf43, 0.9782608691602945)
        buf46 = reinterpret_tensor(buf41, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf41  # reuse
        buf50 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        buf51 = empty((25088, 128), device='cuda', dtype=torch.float32)
        buf818 = empty((8, 16, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div_, mul, x_22, x_28, x_29, x_30], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_10.run(buf46, buf17, buf25, primals_115, primals_119, buf43, primals_8, primals_9, buf50, buf51, buf818, 25088, 128, grid=grid(25088), stream=stream0)
        del primals_115
        del primals_119
        del primals_9
        buf52 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_121, buf51, reinterpret_tensor(primals_120, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf52)
        del primals_121
        buf53 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31, x_34], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_7.run(buf52, buf53, 12845056, grid=grid(12845056), stream=stream0)
        buf54 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf53, reinterpret_tensor(primals_122, (512, 128), (1, 512), 0), out=buf54)
        buf55 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_1], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf55, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf55, 0.9782608691602945)
        buf58 = reinterpret_tensor(buf17, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf17  # reuse
        # Source Nodes: [permute_5], Original ATen: [aten.permute]
        triton_poi_fused_permute_11.run(buf46, buf54, primals_123, buf55, buf58, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        buf59 = reinterpret_tensor(buf54, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf54  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf58, buf59, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf64 = empty((8, 56, 56, 256), device='cuda', dtype=torch.float32)
        buf817 = empty((8, 56, 56, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_13.run(buf60, primals_125, buf64, buf817, 25088, 256, grid=grid(25088), stream=stream0)
        del primals_125
        buf65 = empty_strided((8, 256, 57, 57), (831744, 1, 14592, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_14.run(buf64, primals_10, primals_11, buf65, 6653952, grid=grid(6653952), stream=stream0)
        del primals_11
        buf66 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_47], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_15.run(buf65, buf66, buf67, 2048, 784, grid=grid(2048, 784), stream=stream0)
        buf71 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        buf72 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf816 = empty((8, 4, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv, x_52, y_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_16.run(buf66, primals_12, primals_13, primals_14, buf71, buf72, buf816, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_14
        buf73 = empty((6272, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf72, reinterpret_tensor(primals_126, (256, 768), (1, 256), 0), out=buf73)
        buf74 = empty((8, 8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf73, primals_127, buf74, 1605632, grid=grid(1605632), stream=stream0)
        buf75 = empty((8, 8, 4, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_18.run(buf73, primals_127, buf75, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf76 = empty((256, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf75, (256, 32, 196), (6272, 196, 1), 0), out=buf76)
        buf79 = empty((8, 8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._softmax]
        triton_per_fused__softmax_19.run(buf76, buf79, 50176, 196, grid=grid(50176), stream=stream0)
        buf80 = empty((8, 8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf73, primals_127, buf80, 1605632, grid=grid(1605632), stream=stream0)
        del primals_127
        buf81 = empty((256, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf80, (256, 196, 32), (6272, 32, 1), 0), out=buf81)
        buf82 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.view]
        triton_poi_fused_view_21.run(buf81, buf82, 1605632, grid=grid(1605632), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (6272, 256), (256, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf82, reinterpret_tensor(primals_128, (256, 256), (1, 256), 0), out=buf83)
        buf84 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_2], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf84, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf84, 0.9565217383205891)
        buf87 = reinterpret_tensor(buf83, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf83  # reuse
        buf91 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        buf92 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf815 = empty((8, 4, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__2, mul_2, x_52, x_58, x_59, x_60], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_22.run(buf87, buf66, primals_12, primals_129, buf84, primals_15, primals_16, buf91, buf92, buf815, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_12
        del primals_129
        del primals_16
        buf93 = reinterpret_tensor(buf60, (6272, 1024), (1024, 1), 0); del buf60  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf92, reinterpret_tensor(primals_130, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf93)
        del primals_131
        buf94 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_64], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_23.run(buf93, buf94, 6422528, grid=grid(6422528), stream=stream0)
        buf95 = reinterpret_tensor(buf66, (6272, 256), (256, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf94, reinterpret_tensor(primals_132, (1024, 256), (1, 1024), 0), out=buf95)
        buf96 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_3], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf96, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf96, 0.9565217383205891)
        buf102 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        buf103 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf814 = empty((8, 4, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__3, getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv, mul_3, x_66, y_3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_24.run(buf87, buf95, primals_133, buf96, primals_17, primals_18, buf102, buf103, buf814, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_18
        buf104 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf103, reinterpret_tensor(primals_134, (256, 768), (1, 256), 0), out=buf104)
        buf105 = empty((8, 8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_17.run(buf104, primals_135, buf105, 1605632, grid=grid(1605632), stream=stream0)
        buf106 = empty((8, 8, 4, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_18.run(buf104, primals_135, buf106, 8192, 196, grid=grid(8192, 196), stream=stream0)
        buf107 = buf76; del buf76  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf106, (256, 32, 196), (6272, 196, 1), 0), out=buf107)
        buf110 = empty((8, 8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._softmax]
        triton_per_fused__softmax_19.run(buf107, buf110, 50176, 196, grid=grid(50176), stream=stream0)
        del buf107
        buf111 = empty((8, 8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf104, primals_135, buf111, 1605632, grid=grid(1605632), stream=stream0)
        del buf104
        del primals_135
        buf112 = empty((256, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf111, (256, 196, 32), (6272, 32, 1), 0), out=buf112)
        buf113 = empty((6272, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.view]
        triton_poi_fused_view_21.run(buf112, buf113, 1605632, grid=grid(1605632), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (6272, 256), (256, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf113, reinterpret_tensor(primals_136, (256, 256), (1, 256), 0), out=buf114)
        buf115 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_4], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf115, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf115, 0.9347826093435287)
        buf118 = reinterpret_tensor(buf114, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf114  # reuse
        buf122 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        buf123 = empty((6272, 256), device='cuda', dtype=torch.float32)
        buf813 = empty((8, 4, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__3, div__4, mul_3, mul_4, x_66, x_72, x_73, x_74], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_25.run(buf118, buf87, buf95, primals_133, buf96, primals_137, buf115, primals_19, primals_20, buf122, buf123, buf813, 6272, 256, grid=grid(6272), stream=stream0)
        del primals_133
        del primals_137
        del primals_20
        buf124 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_139, buf123, reinterpret_tensor(primals_138, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf124)
        del primals_139
        buf125 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_75, x_78], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_23.run(buf124, buf125, 6422528, grid=grid(6422528), stream=stream0)
        buf126 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf125, reinterpret_tensor(primals_140, (1024, 256), (1, 1024), 0), out=buf126)
        buf127 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_5], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf127, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf127, 0.9347826093435287)
        buf130 = reinterpret_tensor(buf87, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf87  # reuse
        # Source Nodes: [permute_13], Original ATen: [aten.permute]
        triton_poi_fused_permute_26.run(buf118, buf126, primals_141, buf127, buf130, 1605632, grid=grid(1605632), stream=stream0)
        del buf118
        del primals_141
        buf131 = reinterpret_tensor(buf126, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf126  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf130, buf131, 2048, 784, grid=grid(2048, 784), stream=stream0)
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 512, 28, 28), (401408, 784, 28, 1))
        del buf131
        buf133 = empty_strided((8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((8, 28, 28, 1, 4), (3136, 28, 1, 25088, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_28.run(buf132, primals_143, buf133, buf134, buf135, 25088, 128, grid=grid(25088), stream=stream0)
        buf136 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf812 = empty((8, 28, 28, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_29.run(buf133, buf134, buf135, buf136, buf137, buf812, 6272, 4, grid=grid(6272), stream=stream0)
        del buf133
        del buf134
        del buf135
        buf139 = reinterpret_tensor(buf59, (8, 28, 28, 512), (401408, 14336, 512, 1), 0); del buf59  # reuse
        # Source Nodes: [x_86], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_30.run(buf132, primals_143, buf136, buf137, buf139, 6272, 512, grid=grid(6272, 512), stream=stream0)
        del buf136
        del buf137
        del primals_143
        buf140 = empty_strided((8, 512, 29, 29), (430592, 1, 14848, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_31.run(buf139, primals_21, primals_22, buf140, 3444736, grid=grid(3444736), stream=stream0)
        del primals_22
        buf141 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_91], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_32.run(buf140, buf141, buf142, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf146 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf147 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf811 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv, x_96, y_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33.run(buf141, primals_23, primals_24, primals_25, buf146, buf147, buf811, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_25
        buf148 = empty((1568, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf147, reinterpret_tensor(primals_144, (512, 1536), (1, 512), 0), out=buf148)
        buf149 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf148, primals_145, buf149, 802816, grid=grid(802816), stream=stream0)
        buf150 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf148, primals_145, buf150, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf151 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf150, (128, 32, 196), (6272, 196, 1), 0), out=buf151)
        buf154 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf151, buf154, 25088, 196, grid=grid(25088), stream=stream0)
        buf155 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf148, primals_145, buf155, 802816, grid=grid(802816), stream=stream0)
        del primals_145
        buf156 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf155, (128, 196, 32), (6272, 32, 1), 0), out=buf156)
        buf157 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf156, buf157, 802816, grid=grid(802816), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (1568, 512), (512, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf157, reinterpret_tensor(primals_146, (512, 512), (1, 512), 0), out=buf158)
        buf159 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_6], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf159, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf159, 0.9130434766411781)
        buf162 = reinterpret_tensor(buf158, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf158  # reuse
        buf166 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf167 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf810 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__6, mul_6, x_102, x_103, x_104, x_96], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_39.run(buf162, buf141, primals_23, primals_147, buf159, primals_26, primals_27, buf166, buf167, buf810, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_147
        del primals_23
        del primals_27
        buf168 = reinterpret_tensor(buf132, (1568, 2048), (2048, 1), 0); del buf132  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_149, buf167, reinterpret_tensor(primals_148, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf168)
        del primals_149
        buf169 = reinterpret_tensor(buf46, (1568, 2048), (2048, 1), 0); del buf46  # reuse
        # Source Nodes: [x_105, x_108], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf168, buf169, 3211264, grid=grid(3211264), stream=stream0)
        buf170 = reinterpret_tensor(buf141, (1568, 512), (512, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf169, reinterpret_tensor(primals_150, (2048, 512), (1, 2048), 0), out=buf170)
        buf171 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_7], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf171, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf171, 0.9130434766411781)
        buf177 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf178 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf809 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__7, getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv, mul_7, x_110, y_5], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_41.run(buf162, buf170, primals_151, buf171, primals_28, primals_29, buf177, buf178, buf809, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_29
        buf179 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf178, reinterpret_tensor(primals_152, (512, 1536), (1, 512), 0), out=buf179)
        buf180 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf179, primals_153, buf180, 802816, grid=grid(802816), stream=stream0)
        buf181 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf179, primals_153, buf181, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf182 = buf151; del buf151  # reuse
        # Source Nodes: [x_112], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf181, (128, 32, 196), (6272, 196, 1), 0), out=buf182)
        buf185 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf182, buf185, 25088, 196, grid=grid(25088), stream=stream0)
        buf186 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf179, primals_153, buf186, 802816, grid=grid(802816), stream=stream0)
        del primals_153
        buf187 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf185, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf186, (128, 196, 32), (6272, 32, 1), 0), out=buf187)
        buf188 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_114], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf187, buf188, 802816, grid=grid(802816), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (1568, 512), (512, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf188, reinterpret_tensor(primals_154, (512, 512), (1, 512), 0), out=buf189)
        buf190 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_8], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf190, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf190, 0.8913043439388275)
        buf193 = reinterpret_tensor(buf189, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf189  # reuse
        buf197 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf198 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf808 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__7, div__8, mul_7, mul_8, x_110, x_116, x_117, x_118], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_42.run(buf193, buf162, buf170, primals_151, buf171, primals_155, buf190, primals_30, primals_31, buf197, buf198, buf808, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_151
        del primals_155
        del primals_31
        buf199 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, buf198, reinterpret_tensor(primals_156, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf199)
        del primals_157
        buf200 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_122], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf199, buf200, 3211264, grid=grid(3211264), stream=stream0)
        buf201 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf200, reinterpret_tensor(primals_158, (2048, 512), (1, 2048), 0), out=buf201)
        buf202 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_9], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf202, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf202, 0.8913043439388275)
        buf208 = reinterpret_tensor(buf162, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf162  # reuse
        buf209 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf807 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__9, getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv, mul_9, x_124, y_6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_43.run(buf193, buf201, primals_159, buf202, primals_32, primals_33, buf208, buf209, buf807, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_33
        buf210 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf209, reinterpret_tensor(primals_160, (512, 1536), (1, 512), 0), out=buf210)
        buf211 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf210, primals_161, buf211, 802816, grid=grid(802816), stream=stream0)
        buf212 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf210, primals_161, buf212, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf213 = buf182; del buf182  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf212, (128, 32, 196), (6272, 196, 1), 0), out=buf213)
        buf216 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf213, buf216, 25088, 196, grid=grid(25088), stream=stream0)
        buf217 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf210, primals_161, buf217, 802816, grid=grid(802816), stream=stream0)
        del primals_161
        buf218 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf217, (128, 196, 32), (6272, 32, 1), 0), out=buf218)
        buf219 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf218, buf219, 802816, grid=grid(802816), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (1568, 512), (512, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf219, reinterpret_tensor(primals_162, (512, 512), (1, 512), 0), out=buf220)
        buf221 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_10], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf221, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf221, 0.8695652186870575)
        buf224 = reinterpret_tensor(buf220, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf220  # reuse
        buf228 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf229 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf806 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__10, div__9, mul_10, mul_9, x_124, x_130, x_131, x_132], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_44.run(buf224, buf193, buf201, primals_159, buf202, primals_163, buf221, primals_34, primals_35, buf228, buf229, buf806, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_159
        del primals_163
        del primals_35
        buf230 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_165, buf229, reinterpret_tensor(primals_164, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf230)
        del primals_165
        buf231 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_136], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf230, buf231, 3211264, grid=grid(3211264), stream=stream0)
        buf232 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf231, reinterpret_tensor(primals_166, (2048, 512), (1, 2048), 0), out=buf232)
        buf233 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_11], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf233, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf233, 0.8695652186870575)
        buf239 = reinterpret_tensor(buf193, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf193  # reuse
        buf240 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf805 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__11, getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv, mul_11, x_138, y_7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_45.run(buf224, buf232, primals_167, buf233, primals_36, primals_37, buf239, buf240, buf805, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_37
        buf241 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf240, reinterpret_tensor(primals_168, (512, 1536), (1, 512), 0), out=buf241)
        buf242 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf241, primals_169, buf242, 802816, grid=grid(802816), stream=stream0)
        buf243 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf241, primals_169, buf243, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf244 = buf213; del buf213  # reuse
        # Source Nodes: [x_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf243, (128, 32, 196), (6272, 196, 1), 0), out=buf244)
        buf247 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf244, buf247, 25088, 196, grid=grid(25088), stream=stream0)
        buf248 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf241, primals_169, buf248, 802816, grid=grid(802816), stream=stream0)
        del primals_169
        buf249 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf248, (128, 196, 32), (6272, 32, 1), 0), out=buf249)
        buf250 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf249, buf250, 802816, grid=grid(802816), stream=stream0)
        buf251 = reinterpret_tensor(buf249, (1568, 512), (512, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf250, reinterpret_tensor(primals_170, (512, 512), (1, 512), 0), out=buf251)
        buf252 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_12], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf252, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf252, 0.8478260785341263)
        buf255 = reinterpret_tensor(buf251, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf251  # reuse
        buf259 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf260 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf804 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__11, div__12, mul_11, mul_12, x_138, x_144, x_145, x_146], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_46.run(buf255, buf224, buf232, primals_167, buf233, primals_171, buf252, primals_38, primals_39, buf259, buf260, buf804, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_167
        del primals_171
        del primals_39
        buf261 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_173, buf260, reinterpret_tensor(primals_172, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf261)
        del primals_173
        buf262 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147, x_150], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf261, buf262, 3211264, grid=grid(3211264), stream=stream0)
        buf263 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf262, reinterpret_tensor(primals_174, (2048, 512), (1, 2048), 0), out=buf263)
        buf264 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_13], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf264, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf264, 0.8478260785341263)
        buf270 = reinterpret_tensor(buf224, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf224  # reuse
        buf271 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf803 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__13, getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv, mul_13, x_152, y_8], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_47.run(buf255, buf263, primals_175, buf264, primals_40, primals_41, buf270, buf271, buf803, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_41
        buf272 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf271, reinterpret_tensor(primals_176, (512, 1536), (1, 512), 0), out=buf272)
        buf273 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf272, primals_177, buf273, 802816, grid=grid(802816), stream=stream0)
        buf274 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf272, primals_177, buf274, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf275 = buf244; del buf244  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf273, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf274, (128, 32, 196), (6272, 196, 1), 0), out=buf275)
        buf278 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf275, buf278, 25088, 196, grid=grid(25088), stream=stream0)
        buf279 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf272, primals_177, buf279, 802816, grid=grid(802816), stream=stream0)
        del primals_177
        buf280 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf279, (128, 196, 32), (6272, 32, 1), 0), out=buf280)
        buf281 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf280, buf281, 802816, grid=grid(802816), stream=stream0)
        buf282 = reinterpret_tensor(buf280, (1568, 512), (512, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf281, reinterpret_tensor(primals_178, (512, 512), (1, 512), 0), out=buf282)
        buf283 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_14], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf283, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf283, 0.8260869532823563)
        buf286 = reinterpret_tensor(buf282, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf282  # reuse
        buf290 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf291 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf802 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__13, div__14, mul_13, mul_14, x_152, x_158, x_159, x_160], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_48.run(buf286, buf255, buf263, primals_175, buf264, primals_179, buf283, primals_42, primals_43, buf290, buf291, buf802, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_175
        del primals_179
        del primals_43
        buf292 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_160], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, buf291, reinterpret_tensor(primals_180, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf292)
        del primals_181
        buf293 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161, x_164], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf292, buf293, 3211264, grid=grid(3211264), stream=stream0)
        buf294 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf293, reinterpret_tensor(primals_182, (2048, 512), (1, 2048), 0), out=buf294)
        buf295 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_15], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf295, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf295, 0.8260869532823563)
        buf301 = reinterpret_tensor(buf255, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf255  # reuse
        buf302 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf801 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__15, getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv, mul_15, x_166, y_9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_49.run(buf286, buf294, primals_183, buf295, primals_44, primals_45, buf301, buf302, buf801, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_45
        buf303 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf302, reinterpret_tensor(primals_184, (512, 1536), (1, 512), 0), out=buf303)
        buf304 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf303, primals_185, buf304, 802816, grid=grid(802816), stream=stream0)
        buf305 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf303, primals_185, buf305, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf306 = buf275; del buf275  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf305, (128, 32, 196), (6272, 196, 1), 0), out=buf306)
        buf309 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf306, buf309, 25088, 196, grid=grid(25088), stream=stream0)
        buf310 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf303, primals_185, buf310, 802816, grid=grid(802816), stream=stream0)
        del primals_185
        buf311 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf310, (128, 196, 32), (6272, 32, 1), 0), out=buf311)
        buf312 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf311, buf312, 802816, grid=grid(802816), stream=stream0)
        buf313 = reinterpret_tensor(buf311, (1568, 512), (512, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf312, reinterpret_tensor(primals_186, (512, 512), (1, 512), 0), out=buf313)
        buf314 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_16], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf314, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf314, 0.8043478280305862)
        buf317 = reinterpret_tensor(buf313, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf313  # reuse
        buf321 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf322 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf800 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__15, div__16, mul_15, mul_16, x_166, x_172, x_173, x_174], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_50.run(buf317, buf286, buf294, primals_183, buf295, primals_187, buf314, primals_46, primals_47, buf321, buf322, buf800, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_183
        del primals_187
        del primals_47
        buf323 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_189, buf322, reinterpret_tensor(primals_188, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf323)
        del primals_189
        buf324 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_178], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf323, buf324, 3211264, grid=grid(3211264), stream=stream0)
        buf325 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf324, reinterpret_tensor(primals_190, (2048, 512), (1, 2048), 0), out=buf325)
        buf326 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_17], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf326, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf326, 0.8043478280305862)
        buf332 = reinterpret_tensor(buf286, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf286  # reuse
        buf333 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf799 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__17, getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv, mul_17, x_180, y_10], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_51.run(buf317, buf325, primals_191, buf326, primals_48, primals_49, buf332, buf333, buf799, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_49
        buf334 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf333, reinterpret_tensor(primals_192, (512, 1536), (1, 512), 0), out=buf334)
        buf335 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf334, primals_193, buf335, 802816, grid=grid(802816), stream=stream0)
        buf336 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf334, primals_193, buf336, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf337 = buf306; del buf306  # reuse
        # Source Nodes: [x_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf336, (128, 32, 196), (6272, 196, 1), 0), out=buf337)
        buf340 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf337, buf340, 25088, 196, grid=grid(25088), stream=stream0)
        buf341 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf334, primals_193, buf341, 802816, grid=grid(802816), stream=stream0)
        del primals_193
        buf342 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf341, (128, 196, 32), (6272, 32, 1), 0), out=buf342)
        buf343 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf342, buf343, 802816, grid=grid(802816), stream=stream0)
        buf344 = reinterpret_tensor(buf342, (1568, 512), (512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf343, reinterpret_tensor(primals_194, (512, 512), (1, 512), 0), out=buf344)
        buf345 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_18], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf345, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf345, 0.782608687877655)
        buf348 = reinterpret_tensor(buf344, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf344  # reuse
        buf352 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf353 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf798 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__17, div__18, mul_17, mul_18, x_180, x_186, x_187, x_188], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_52.run(buf348, buf317, buf325, primals_191, buf326, primals_195, buf345, primals_50, primals_51, buf352, buf353, buf798, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_191
        del primals_195
        del primals_51
        buf354 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_197, buf353, reinterpret_tensor(primals_196, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf354)
        del primals_197
        buf355 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_189, x_192], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf354, buf355, 3211264, grid=grid(3211264), stream=stream0)
        buf356 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf355, reinterpret_tensor(primals_198, (2048, 512), (1, 2048), 0), out=buf356)
        buf357 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_19], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf357, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf357, 0.782608687877655)
        buf363 = reinterpret_tensor(buf317, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf317  # reuse
        buf364 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf797 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__19, getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv, mul_19, x_194, y_11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_53.run(buf348, buf356, primals_199, buf357, primals_52, primals_53, buf363, buf364, buf797, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_53
        buf365 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf364, reinterpret_tensor(primals_200, (512, 1536), (1, 512), 0), out=buf365)
        buf366 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf365, primals_201, buf366, 802816, grid=grid(802816), stream=stream0)
        buf367 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf365, primals_201, buf367, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf368 = buf337; del buf337  # reuse
        # Source Nodes: [x_196], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf367, (128, 32, 196), (6272, 196, 1), 0), out=buf368)
        buf371 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf368, buf371, 25088, 196, grid=grid(25088), stream=stream0)
        buf372 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf365, primals_201, buf372, 802816, grid=grid(802816), stream=stream0)
        del primals_201
        buf373 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_196], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf371, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf372, (128, 196, 32), (6272, 32, 1), 0), out=buf373)
        buf374 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf373, buf374, 802816, grid=grid(802816), stream=stream0)
        buf375 = reinterpret_tensor(buf373, (1568, 512), (512, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf374, reinterpret_tensor(primals_202, (512, 512), (1, 512), 0), out=buf375)
        buf376 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_20], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf376, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf376, 0.760869562625885)
        buf379 = reinterpret_tensor(buf375, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf375  # reuse
        buf383 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf384 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf796 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__19, div__20, mul_19, mul_20, x_194, x_200, x_201, x_202], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_54.run(buf379, buf348, buf356, primals_199, buf357, primals_203, buf376, primals_54, primals_55, buf383, buf384, buf796, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_199
        del primals_203
        del primals_55
        buf385 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, buf384, reinterpret_tensor(primals_204, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf385)
        del primals_205
        buf386 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203, x_206], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf385, buf386, 3211264, grid=grid(3211264), stream=stream0)
        buf387 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf386, reinterpret_tensor(primals_206, (2048, 512), (1, 2048), 0), out=buf387)
        buf388 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_21], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf388, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf388, 0.760869562625885)
        buf394 = reinterpret_tensor(buf348, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf348  # reuse
        buf395 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf795 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__21, getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv, mul_21, x_208, y_12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_55.run(buf379, buf387, primals_207, buf388, primals_56, primals_57, buf394, buf395, buf795, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_57
        buf396 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf395, reinterpret_tensor(primals_208, (512, 1536), (1, 512), 0), out=buf396)
        buf397 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf396, primals_209, buf397, 802816, grid=grid(802816), stream=stream0)
        buf398 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf396, primals_209, buf398, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf399 = buf368; del buf368  # reuse
        # Source Nodes: [x_210], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf397, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf398, (128, 32, 196), (6272, 196, 1), 0), out=buf399)
        buf402 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf399, buf402, 25088, 196, grid=grid(25088), stream=stream0)
        buf403 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf396, primals_209, buf403, 802816, grid=grid(802816), stream=stream0)
        del primals_209
        buf404 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_210], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf403, (128, 196, 32), (6272, 32, 1), 0), out=buf404)
        buf405 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf404, buf405, 802816, grid=grid(802816), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (1568, 512), (512, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf405, reinterpret_tensor(primals_210, (512, 512), (1, 512), 0), out=buf406)
        buf407 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_22], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf407, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf407, 0.739130437374115)
        buf410 = reinterpret_tensor(buf406, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf406  # reuse
        buf414 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf415 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf794 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__21, div__22, mul_21, mul_22, x_208, x_214, x_215, x_216], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_56.run(buf410, buf379, buf387, primals_207, buf388, primals_211, buf407, primals_58, primals_59, buf414, buf415, buf794, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_207
        del primals_211
        del primals_59
        buf416 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, buf415, reinterpret_tensor(primals_212, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf416)
        del primals_213
        buf417 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217, x_220], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf416, buf417, 3211264, grid=grid(3211264), stream=stream0)
        buf418 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf417, reinterpret_tensor(primals_214, (2048, 512), (1, 2048), 0), out=buf418)
        buf419 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_23], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf419, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf419, 0.739130437374115)
        buf425 = reinterpret_tensor(buf379, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf379  # reuse
        buf426 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf793 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__23, getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv, mul_23, x_222, y_13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_57.run(buf410, buf418, primals_215, buf419, primals_60, primals_61, buf425, buf426, buf793, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_61
        buf427 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf426, reinterpret_tensor(primals_216, (512, 1536), (1, 512), 0), out=buf427)
        buf428 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf427, primals_217, buf428, 802816, grid=grid(802816), stream=stream0)
        buf429 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf427, primals_217, buf429, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf430 = buf399; del buf399  # reuse
        # Source Nodes: [x_224], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf429, (128, 32, 196), (6272, 196, 1), 0), out=buf430)
        buf433 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf430, buf433, 25088, 196, grid=grid(25088), stream=stream0)
        buf434 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf427, primals_217, buf434, 802816, grid=grid(802816), stream=stream0)
        del primals_217
        buf435 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf433, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf434, (128, 196, 32), (6272, 32, 1), 0), out=buf435)
        buf436 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf435, buf436, 802816, grid=grid(802816), stream=stream0)
        buf437 = reinterpret_tensor(buf435, (1568, 512), (512, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf436, reinterpret_tensor(primals_218, (512, 512), (1, 512), 0), out=buf437)
        buf438 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_24], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf438, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf438, 0.717391312122345)
        buf441 = reinterpret_tensor(buf437, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf437  # reuse
        buf445 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf446 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf792 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__23, div__24, mul_23, mul_24, x_222, x_228, x_229, x_230], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_58.run(buf441, buf410, buf418, primals_215, buf419, primals_219, buf438, primals_62, primals_63, buf445, buf446, buf792, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_215
        del primals_219
        del primals_63
        buf447 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_221, buf446, reinterpret_tensor(primals_220, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf447)
        del primals_221
        buf448 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231, x_234], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf447, buf448, 3211264, grid=grid(3211264), stream=stream0)
        buf449 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf448, reinterpret_tensor(primals_222, (2048, 512), (1, 2048), 0), out=buf449)
        buf450 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_25], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf450, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf450, 0.717391312122345)
        buf456 = reinterpret_tensor(buf410, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf410  # reuse
        buf457 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf791 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__25, getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv, mul_25, x_236, y_14], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_59.run(buf441, buf449, primals_223, buf450, primals_64, primals_65, buf456, buf457, buf791, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_65
        buf458 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf457, reinterpret_tensor(primals_224, (512, 1536), (1, 512), 0), out=buf458)
        buf459 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf458, primals_225, buf459, 802816, grid=grid(802816), stream=stream0)
        buf460 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf458, primals_225, buf460, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf461 = buf430; del buf430  # reuse
        # Source Nodes: [x_238], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf459, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf460, (128, 32, 196), (6272, 196, 1), 0), out=buf461)
        buf464 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf461, buf464, 25088, 196, grid=grid(25088), stream=stream0)
        buf465 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf458, primals_225, buf465, 802816, grid=grid(802816), stream=stream0)
        del primals_225
        buf466 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf464, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf465, (128, 196, 32), (6272, 32, 1), 0), out=buf466)
        buf467 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_240], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf466, buf467, 802816, grid=grid(802816), stream=stream0)
        buf468 = reinterpret_tensor(buf466, (1568, 512), (512, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf467, reinterpret_tensor(primals_226, (512, 512), (1, 512), 0), out=buf468)
        buf469 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_26], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf469, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf469, 0.695652186870575)
        buf472 = reinterpret_tensor(buf468, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf468  # reuse
        buf476 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf477 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf790 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__25, div__26, mul_25, mul_26, x_236, x_242, x_243, x_244], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_60.run(buf472, buf441, buf449, primals_223, buf450, primals_227, buf469, primals_66, primals_67, buf476, buf477, buf790, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_223
        del primals_227
        del primals_67
        buf478 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_244], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_229, buf477, reinterpret_tensor(primals_228, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf478)
        del primals_229
        buf479 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245, x_248], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf478, buf479, 3211264, grid=grid(3211264), stream=stream0)
        buf480 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf479, reinterpret_tensor(primals_230, (2048, 512), (1, 2048), 0), out=buf480)
        buf481 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_27], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf481, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf481, 0.695652186870575)
        buf487 = reinterpret_tensor(buf441, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf441  # reuse
        buf488 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf789 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__27, getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv, mul_27, x_250, y_15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_61.run(buf472, buf480, primals_231, buf481, primals_68, primals_69, buf487, buf488, buf789, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_69
        buf489 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf488, reinterpret_tensor(primals_232, (512, 1536), (1, 512), 0), out=buf489)
        buf490 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf489, primals_233, buf490, 802816, grid=grid(802816), stream=stream0)
        buf491 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf489, primals_233, buf491, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf492 = buf461; del buf461  # reuse
        # Source Nodes: [x_252], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf490, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf491, (128, 32, 196), (6272, 196, 1), 0), out=buf492)
        buf495 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf492, buf495, 25088, 196, grid=grid(25088), stream=stream0)
        buf496 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf489, primals_233, buf496, 802816, grid=grid(802816), stream=stream0)
        del primals_233
        buf497 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_252], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf496, (128, 196, 32), (6272, 32, 1), 0), out=buf497)
        buf498 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf497, buf498, 802816, grid=grid(802816), stream=stream0)
        buf499 = reinterpret_tensor(buf497, (1568, 512), (512, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf498, reinterpret_tensor(primals_234, (512, 512), (1, 512), 0), out=buf499)
        buf500 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_28], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf500, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf500, 0.6739130616188049)
        buf503 = reinterpret_tensor(buf499, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf499  # reuse
        buf507 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf508 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf788 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__27, div__28, mul_27, mul_28, x_250, x_256, x_257, x_258], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_62.run(buf503, buf472, buf480, primals_231, buf481, primals_235, buf500, primals_70, primals_71, buf507, buf508, buf788, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_231
        del primals_235
        del primals_71
        buf509 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_258], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_237, buf508, reinterpret_tensor(primals_236, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf509)
        del primals_237
        buf510 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259, x_262], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf509, buf510, 3211264, grid=grid(3211264), stream=stream0)
        buf511 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf510, reinterpret_tensor(primals_238, (2048, 512), (1, 2048), 0), out=buf511)
        buf512 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_29], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf512, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf512, 0.6739130616188049)
        buf518 = reinterpret_tensor(buf472, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf472  # reuse
        buf519 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf787 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__29, getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv, mul_29, x_264, y_16], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_63.run(buf503, buf511, primals_239, buf512, primals_72, primals_73, buf518, buf519, buf787, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_73
        buf520 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf519, reinterpret_tensor(primals_240, (512, 1536), (1, 512), 0), out=buf520)
        buf521 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf520, primals_241, buf521, 802816, grid=grid(802816), stream=stream0)
        buf522 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf520, primals_241, buf522, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf523 = buf492; del buf492  # reuse
        # Source Nodes: [x_266], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf521, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf522, (128, 32, 196), (6272, 196, 1), 0), out=buf523)
        buf526 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf523, buf526, 25088, 196, grid=grid(25088), stream=stream0)
        buf527 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf520, primals_241, buf527, 802816, grid=grid(802816), stream=stream0)
        del primals_241
        buf528 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf526, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf527, (128, 196, 32), (6272, 32, 1), 0), out=buf528)
        buf529 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_268], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf528, buf529, 802816, grid=grid(802816), stream=stream0)
        buf530 = reinterpret_tensor(buf528, (1568, 512), (512, 1), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf529, reinterpret_tensor(primals_242, (512, 512), (1, 512), 0), out=buf530)
        buf531 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_30], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf531, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf531, 0.6521739065647125)
        buf534 = reinterpret_tensor(buf530, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf530  # reuse
        buf538 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf539 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf786 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__29, div__30, mul_29, mul_30, x_264, x_270, x_271, x_272], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_64.run(buf534, buf503, buf511, primals_239, buf512, primals_243, buf531, primals_74, primals_75, buf538, buf539, buf786, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_239
        del primals_243
        del primals_75
        buf540 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_245, buf539, reinterpret_tensor(primals_244, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf540)
        del primals_245
        buf541 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_273, x_276], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf540, buf541, 3211264, grid=grid(3211264), stream=stream0)
        buf542 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf541, reinterpret_tensor(primals_246, (2048, 512), (1, 2048), 0), out=buf542)
        buf543 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_31], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf543, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf543, 0.6521739065647125)
        buf549 = reinterpret_tensor(buf503, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf503  # reuse
        buf550 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf785 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__31, getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv, mul_31, x_278, y_17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_65.run(buf534, buf542, primals_247, buf543, primals_76, primals_77, buf549, buf550, buf785, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_77
        buf551 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf550, reinterpret_tensor(primals_248, (512, 1536), (1, 512), 0), out=buf551)
        buf552 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf551, primals_249, buf552, 802816, grid=grid(802816), stream=stream0)
        buf553 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf551, primals_249, buf553, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf554 = buf523; del buf523  # reuse
        # Source Nodes: [x_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf553, (128, 32, 196), (6272, 196, 1), 0), out=buf554)
        buf557 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf554, buf557, 25088, 196, grid=grid(25088), stream=stream0)
        buf558 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf551, primals_249, buf558, 802816, grid=grid(802816), stream=stream0)
        del primals_249
        buf559 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_280], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf558, (128, 196, 32), (6272, 32, 1), 0), out=buf559)
        buf560 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_282], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf559, buf560, 802816, grid=grid(802816), stream=stream0)
        buf561 = reinterpret_tensor(buf559, (1568, 512), (512, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf560, reinterpret_tensor(primals_250, (512, 512), (1, 512), 0), out=buf561)
        buf562 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_32], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf562, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf562, 0.6304347813129425)
        buf565 = reinterpret_tensor(buf561, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf561  # reuse
        buf569 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf570 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf784 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__31, div__32, mul_31, mul_32, x_278, x_284, x_285, x_286], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_66.run(buf565, buf534, buf542, primals_247, buf543, primals_251, buf562, primals_78, primals_79, buf569, buf570, buf784, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_247
        del primals_251
        del primals_79
        buf571 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_253, buf570, reinterpret_tensor(primals_252, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf571)
        del primals_253
        buf572 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287, x_290], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf571, buf572, 3211264, grid=grid(3211264), stream=stream0)
        buf573 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf572, reinterpret_tensor(primals_254, (2048, 512), (1, 2048), 0), out=buf573)
        buf574 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_33], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf574, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf574, 0.6304347813129425)
        buf580 = reinterpret_tensor(buf534, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf534  # reuse
        buf581 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf783 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__33, getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv, mul_33, x_292, y_18], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_67.run(buf565, buf573, primals_255, buf574, primals_80, primals_81, buf580, buf581, buf783, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_81
        buf582 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf581, reinterpret_tensor(primals_256, (512, 1536), (1, 512), 0), out=buf582)
        buf583 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf582, primals_257, buf583, 802816, grid=grid(802816), stream=stream0)
        buf584 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf582, primals_257, buf584, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf585 = buf554; del buf554  # reuse
        # Source Nodes: [x_294], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf583, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf584, (128, 32, 196), (6272, 196, 1), 0), out=buf585)
        buf588 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf585, buf588, 25088, 196, grid=grid(25088), stream=stream0)
        buf589 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf582, primals_257, buf589, 802816, grid=grid(802816), stream=stream0)
        del primals_257
        buf590 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_294], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf589, (128, 196, 32), (6272, 32, 1), 0), out=buf590)
        buf591 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf590, buf591, 802816, grid=grid(802816), stream=stream0)
        buf592 = reinterpret_tensor(buf590, (1568, 512), (512, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf591, reinterpret_tensor(primals_258, (512, 512), (1, 512), 0), out=buf592)
        buf593 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_34], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf593, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf593, 0.6086956560611725)
        buf596 = reinterpret_tensor(buf592, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf592  # reuse
        buf600 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf601 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf782 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__33, div__34, mul_33, mul_34, x_292, x_298, x_299, x_300], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_68.run(buf596, buf565, buf573, primals_255, buf574, primals_259, buf593, primals_82, primals_83, buf600, buf601, buf782, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_255
        del primals_259
        del primals_83
        buf602 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_261, buf601, reinterpret_tensor(primals_260, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf602)
        del primals_261
        buf603 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301, x_304], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf602, buf603, 3211264, grid=grid(3211264), stream=stream0)
        buf604 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf603, reinterpret_tensor(primals_262, (2048, 512), (1, 2048), 0), out=buf604)
        buf605 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_35], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf605, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf605, 0.6086956560611725)
        buf611 = reinterpret_tensor(buf565, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf565  # reuse
        buf612 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf781 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__35, getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv, mul_35, x_306, y_19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_69.run(buf596, buf604, primals_263, buf605, primals_84, primals_85, buf611, buf612, buf781, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_85
        buf613 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf612, reinterpret_tensor(primals_264, (512, 1536), (1, 512), 0), out=buf613)
        buf614 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf613, primals_265, buf614, 802816, grid=grid(802816), stream=stream0)
        buf615 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf613, primals_265, buf615, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf616 = buf585; del buf585  # reuse
        # Source Nodes: [x_308], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf614, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf615, (128, 32, 196), (6272, 196, 1), 0), out=buf616)
        buf619 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf616, buf619, 25088, 196, grid=grid(25088), stream=stream0)
        buf620 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf613, primals_265, buf620, 802816, grid=grid(802816), stream=stream0)
        del primals_265
        buf621 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf619, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf620, (128, 196, 32), (6272, 32, 1), 0), out=buf621)
        buf622 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf621, buf622, 802816, grid=grid(802816), stream=stream0)
        buf623 = reinterpret_tensor(buf621, (1568, 512), (512, 1), 0); del buf621  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf622, reinterpret_tensor(primals_266, (512, 512), (1, 512), 0), out=buf623)
        buf624 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_36], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf624, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf624, 0.5869565308094025)
        buf627 = reinterpret_tensor(buf623, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf623  # reuse
        buf631 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf632 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf780 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__35, div__36, mul_35, mul_36, x_306, x_312, x_313, x_314], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_70.run(buf627, buf596, buf604, primals_263, buf605, primals_267, buf624, primals_86, primals_87, buf631, buf632, buf780, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_263
        del primals_267
        del primals_87
        buf633 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_269, buf632, reinterpret_tensor(primals_268, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf633)
        del primals_269
        buf634 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315, x_318], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf633, buf634, 3211264, grid=grid(3211264), stream=stream0)
        buf635 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf634, reinterpret_tensor(primals_270, (2048, 512), (1, 2048), 0), out=buf635)
        buf636 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_37], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf636, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf636, 0.5869565308094025)
        buf642 = reinterpret_tensor(buf596, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf596  # reuse
        buf643 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf779 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__37, getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv, mul_37, x_320, y_20], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_71.run(buf627, buf635, primals_271, buf636, primals_88, primals_89, buf642, buf643, buf779, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_89
        buf644 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf643, reinterpret_tensor(primals_272, (512, 1536), (1, 512), 0), out=buf644)
        buf645 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf644, primals_273, buf645, 802816, grid=grid(802816), stream=stream0)
        buf646 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf644, primals_273, buf646, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf647 = buf616; del buf616  # reuse
        # Source Nodes: [x_322], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf645, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf646, (128, 32, 196), (6272, 196, 1), 0), out=buf647)
        buf650 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf647, buf650, 25088, 196, grid=grid(25088), stream=stream0)
        buf651 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf644, primals_273, buf651, 802816, grid=grid(802816), stream=stream0)
        del primals_273
        buf652 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf650, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf651, (128, 196, 32), (6272, 32, 1), 0), out=buf652)
        buf653 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf652, buf653, 802816, grid=grid(802816), stream=stream0)
        buf654 = reinterpret_tensor(buf652, (1568, 512), (512, 1), 0); del buf652  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf653, reinterpret_tensor(primals_274, (512, 512), (1, 512), 0), out=buf654)
        buf655 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_38], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf655, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf655, 0.5652174055576324)
        buf658 = reinterpret_tensor(buf654, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf654  # reuse
        buf662 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf663 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf778 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__37, div__38, mul_37, mul_38, x_320, x_326, x_327, x_328], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_72.run(buf658, buf627, buf635, primals_271, buf636, primals_275, buf655, primals_90, primals_91, buf662, buf663, buf778, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_271
        del primals_275
        del primals_91
        buf664 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_328], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_277, buf663, reinterpret_tensor(primals_276, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf664)
        del primals_277
        buf665 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329, x_332], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf664, buf665, 3211264, grid=grid(3211264), stream=stream0)
        buf666 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf665, reinterpret_tensor(primals_278, (2048, 512), (1, 2048), 0), out=buf666)
        buf667 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_39], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf667, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf667, 0.5652174055576324)
        buf673 = reinterpret_tensor(buf627, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf627  # reuse
        buf674 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf777 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__39, getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv, mul_39, x_334, y_21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_73.run(buf658, buf666, primals_279, buf667, primals_92, primals_93, buf673, buf674, buf777, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_93
        buf675 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf674, reinterpret_tensor(primals_280, (512, 1536), (1, 512), 0), out=buf675)
        buf676 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf675, primals_281, buf676, 802816, grid=grid(802816), stream=stream0)
        buf677 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf675, primals_281, buf677, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf678 = buf647; del buf647  # reuse
        # Source Nodes: [x_336], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf676, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf677, (128, 32, 196), (6272, 196, 1), 0), out=buf678)
        buf681 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf678, buf681, 25088, 196, grid=grid(25088), stream=stream0)
        buf682 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf675, primals_281, buf682, 802816, grid=grid(802816), stream=stream0)
        del primals_281
        buf683 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_336], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf681, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf682, (128, 196, 32), (6272, 32, 1), 0), out=buf683)
        buf684 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_338], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf683, buf684, 802816, grid=grid(802816), stream=stream0)
        buf685 = reinterpret_tensor(buf683, (1568, 512), (512, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf684, reinterpret_tensor(primals_282, (512, 512), (1, 512), 0), out=buf685)
        buf686 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_40], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf686, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf686, 0.54347825050354)
        buf689 = reinterpret_tensor(buf685, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf685  # reuse
        buf693 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf694 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf776 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__39, div__40, mul_39, mul_40, x_334, x_340, x_341, x_342], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_74.run(buf689, buf658, buf666, primals_279, buf667, primals_283, buf686, primals_94, primals_95, buf693, buf694, buf776, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_279
        del primals_283
        del primals_95
        buf695 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_342], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_285, buf694, reinterpret_tensor(primals_284, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf695)
        del primals_285
        buf696 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343, x_346], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf695, buf696, 3211264, grid=grid(3211264), stream=stream0)
        buf697 = buf666; del buf666  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf696, reinterpret_tensor(primals_286, (2048, 512), (1, 2048), 0), out=buf697)
        buf698 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_41], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf698, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf698, 0.54347825050354)
        buf704 = reinterpret_tensor(buf658, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf658  # reuse
        buf705 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf775 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__41, getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv, mul_41, x_348, y_22], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_75.run(buf689, buf697, primals_287, buf698, primals_96, primals_97, buf704, buf705, buf775, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_97
        buf706 = buf675; del buf675  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf705, reinterpret_tensor(primals_288, (512, 1536), (1, 512), 0), out=buf706)
        buf707 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf706, primals_289, buf707, 802816, grid=grid(802816), stream=stream0)
        buf708 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf706, primals_289, buf708, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf709 = buf678; del buf678  # reuse
        # Source Nodes: [x_350], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf707, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf708, (128, 32, 196), (6272, 196, 1), 0), out=buf709)
        buf712 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf709, buf712, 25088, 196, grid=grid(25088), stream=stream0)
        buf713 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf706, primals_289, buf713, 802816, grid=grid(802816), stream=stream0)
        del primals_289
        buf714 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf712, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf713, (128, 196, 32), (6272, 32, 1), 0), out=buf714)
        buf715 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_352], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf714, buf715, 802816, grid=grid(802816), stream=stream0)
        buf716 = reinterpret_tensor(buf714, (1568, 512), (512, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf715, reinterpret_tensor(primals_290, (512, 512), (1, 512), 0), out=buf716)
        buf717 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_42], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf717, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf717, 0.52173912525177)
        buf720 = reinterpret_tensor(buf716, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf716  # reuse
        buf724 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf725 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf774 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__41, div__42, mul_41, mul_42, x_348, x_354, x_355, x_356], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_76.run(buf720, buf689, buf697, primals_287, buf698, primals_291, buf717, primals_98, primals_99, buf724, buf725, buf774, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_287
        del primals_291
        del primals_99
        buf726 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_356], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_293, buf725, reinterpret_tensor(primals_292, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf726)
        del primals_293
        buf727 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_357, x_360], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf726, buf727, 3211264, grid=grid(3211264), stream=stream0)
        buf728 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf727, reinterpret_tensor(primals_294, (2048, 512), (1, 2048), 0), out=buf728)
        buf729 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_43], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf729, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf729, 0.52173912525177)
        buf735 = reinterpret_tensor(buf689, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf689  # reuse
        buf736 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf773 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__43, getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv, mul_43, x_362, y_23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_77.run(buf720, buf728, primals_295, buf729, primals_100, primals_101, buf735, buf736, buf773, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_101
        buf737 = buf706; del buf706  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf736, reinterpret_tensor(primals_296, (512, 1536), (1, 512), 0), out=buf737)
        buf738 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_34.run(buf737, primals_297, buf738, 802816, grid=grid(802816), stream=stream0)
        buf739 = empty((8, 16, 1, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_35.run(buf737, primals_297, buf739, 4096, 196, grid=grid(4096, 196), stream=stream0)
        buf740 = buf709; del buf709  # reuse
        # Source Nodes: [x_364], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf738, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf739, (128, 32, 196), (6272, 196, 1), 0), out=buf740)
        buf743 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten._softmax]
        triton_per_fused__softmax_36.run(buf740, buf743, 25088, 196, grid=grid(25088), stream=stream0)
        del buf740
        buf744 = empty((8, 16, 1, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf737, primals_297, buf744, 802816, grid=grid(802816), stream=stream0)
        del buf737
        del primals_297
        buf745 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_364], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf743, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf744, (128, 196, 32), (6272, 32, 1), 0), out=buf745)
        buf746 = empty((1568, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_366], Original ATen: [aten.view]
        triton_poi_fused_view_38.run(buf745, buf746, 802816, grid=grid(802816), stream=stream0)
        buf747 = reinterpret_tensor(buf745, (1568, 512), (512, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf746, reinterpret_tensor(primals_298, (512, 512), (1, 512), 0), out=buf747)
        buf748 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_44], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf748, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf748, 0.5)
        buf751 = reinterpret_tensor(buf747, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf747  # reuse
        buf755 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        buf756 = empty((1568, 512), device='cuda', dtype=torch.float32)
        buf772 = empty((8, 1, 196, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__43, div__44, mul_43, mul_44, x_362, x_368, x_369, x_370], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_view_78.run(buf751, buf720, buf728, primals_295, buf729, primals_299, buf748, primals_102, primals_103, buf755, buf756, buf772, 1568, 512, grid=grid(1568), stream=stream0)
        del primals_103
        del primals_295
        del primals_299
        buf757 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_370], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_301, buf756, reinterpret_tensor(primals_300, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf757)
        del primals_301
        buf758 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371, x_374], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_40.run(buf757, buf758, 3211264, grid=grid(3211264), stream=stream0)
        buf759 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf758, reinterpret_tensor(primals_302, (2048, 512), (1, 2048), 0), out=buf759)
        buf760 = empty((8, 1, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [random_tensor_45], Original ATen: [aten.bernoulli]
        triton_poi_fused_bernoulli_9.run(buf760, 8, grid=grid(8), stream=stream0)
        aten.bernoulli_(buf760, 0.5)
        buf766 = reinterpret_tensor(buf720, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf720  # reuse
        buf771 = empty((8, 14, 14, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_79.run(buf751, buf759, primals_303, buf760, buf766, buf771, 1568, 512, grid=grid(1568), stream=stream0)
        del buf751
        del buf759
        del primals_303
        buf767 = empty_strided((8, 512, 1, 1, 2), (1024, 1, 8192, 8192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_385], Original ATen: [aten.mean]
        triton_red_fused_mean_80.run(buf766, primals_104, primals_105, buf767, 8192, 98, grid=grid(8192), stream=stream0)
        del primals_105
        buf768 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf769 = reinterpret_tensor(buf768, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf768  # reuse
        # Source Nodes: [x_385], Original ATen: [aten.mean]
        triton_per_fused_mean_81.run(buf769, buf767, 4096, 2, grid=grid(4096), stream=stream0)
        del buf767
        buf770 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_305, reinterpret_tensor(buf769, (8, 512), (512, 1), 0), reinterpret_tensor(primals_304, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf770)
        del primals_305
        return (buf770, primals_2, primals_4, primals_6, primals_8, primals_10, primals_13, primals_15, primals_17, primals_19, primals_21, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_124, primals_142, primals_306, buf4, buf5, buf15, buf21, buf22, buf23, buf24, buf29, buf30, buf40, buf43, buf50, buf51, buf52, buf53, buf55, buf58, buf64, buf65, buf67, buf71, buf72, buf82, buf84, buf91, buf92, buf93, buf94, buf96, buf102, buf103, buf113, buf115, buf122, buf123, buf124, buf125, buf127, buf130, buf139, buf140, buf142, buf146, buf147, buf157, buf159, buf166, buf167, buf168, buf169, buf171, buf177, buf178, buf188, buf190, buf197, buf198, buf199, buf200, buf202, buf208, buf209, buf219, buf221, buf228, buf229, buf230, buf231, buf233, buf239, buf240, buf250, buf252, buf259, buf260, buf261, buf262, buf264, buf270, buf271, buf281, buf283, buf290, buf291, buf292, buf293, buf295, buf301, buf302, buf312, buf314, buf321, buf322, buf323, buf324, buf326, buf332, buf333, buf343, buf345, buf352, buf353, buf354, buf355, buf357, buf363, buf364, buf374, buf376, buf383, buf384, buf385, buf386, buf388, buf394, buf395, buf405, buf407, buf414, buf415, buf416, buf417, buf419, buf425, buf426, buf436, buf438, buf445, buf446, buf447, buf448, buf450, buf456, buf457, buf467, buf469, buf476, buf477, buf478, buf479, buf481, buf487, buf488, buf498, buf500, buf507, buf508, buf509, buf510, buf512, buf518, buf519, buf529, buf531, buf538, buf539, buf540, buf541, buf543, buf549, buf550, buf560, buf562, buf569, buf570, buf571, buf572, buf574, buf580, buf581, buf591, buf593, buf600, buf601, buf602, buf603, buf605, buf611, buf612, buf622, buf624, buf631, buf632, buf633, buf634, buf636, buf642, buf643, buf653, buf655, buf662, buf663, buf664, buf665, buf667, buf673, buf674, buf684, buf686, buf693, buf694, buf695, buf696, buf698, buf704, buf705, buf715, buf717, buf724, buf725, buf726, buf727, buf729, buf735, buf736, buf746, buf748, buf755, buf756, buf757, buf758, buf760, buf766, reinterpret_tensor(buf769, (8, 512), (512, 1), 0), reinterpret_tensor(primals_304, (1000, 512), (512, 1), 0), buf771, reinterpret_tensor(primals_302, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_300, (2048, 512), (512, 1), 0), buf772, reinterpret_tensor(primals_298, (512, 512), (512, 1), 0), reinterpret_tensor(buf743, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf744, (128, 32, 196), (6272, 1, 32), 0), buf743, reinterpret_tensor(buf738, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf739, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_296, (1536, 512), (512, 1), 0), buf773, reinterpret_tensor(primals_294, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_292, (2048, 512), (512, 1), 0), buf774, reinterpret_tensor(primals_290, (512, 512), (512, 1), 0), reinterpret_tensor(buf712, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf713, (128, 32, 196), (6272, 1, 32), 0), buf712, reinterpret_tensor(buf707, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf708, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_288, (1536, 512), (512, 1), 0), buf775, reinterpret_tensor(primals_286, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_284, (2048, 512), (512, 1), 0), buf776, reinterpret_tensor(primals_282, (512, 512), (512, 1), 0), reinterpret_tensor(buf681, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf682, (128, 32, 196), (6272, 1, 32), 0), buf681, reinterpret_tensor(buf676, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf677, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_280, (1536, 512), (512, 1), 0), buf777, reinterpret_tensor(primals_278, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_276, (2048, 512), (512, 1), 0), buf778, reinterpret_tensor(primals_274, (512, 512), (512, 1), 0), reinterpret_tensor(buf650, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf651, (128, 32, 196), (6272, 1, 32), 0), buf650, reinterpret_tensor(buf645, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf646, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_272, (1536, 512), (512, 1), 0), buf779, reinterpret_tensor(primals_270, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_268, (2048, 512), (512, 1), 0), buf780, reinterpret_tensor(primals_266, (512, 512), (512, 1), 0), reinterpret_tensor(buf619, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf620, (128, 32, 196), (6272, 1, 32), 0), buf619, reinterpret_tensor(buf614, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf615, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_264, (1536, 512), (512, 1), 0), buf781, reinterpret_tensor(primals_262, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_260, (2048, 512), (512, 1), 0), buf782, reinterpret_tensor(primals_258, (512, 512), (512, 1), 0), reinterpret_tensor(buf588, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf589, (128, 32, 196), (6272, 1, 32), 0), buf588, reinterpret_tensor(buf583, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf584, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_256, (1536, 512), (512, 1), 0), buf783, reinterpret_tensor(primals_254, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_252, (2048, 512), (512, 1), 0), buf784, reinterpret_tensor(primals_250, (512, 512), (512, 1), 0), reinterpret_tensor(buf557, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf558, (128, 32, 196), (6272, 1, 32), 0), buf557, reinterpret_tensor(buf552, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf553, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_248, (1536, 512), (512, 1), 0), buf785, reinterpret_tensor(primals_246, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_244, (2048, 512), (512, 1), 0), buf786, reinterpret_tensor(primals_242, (512, 512), (512, 1), 0), reinterpret_tensor(buf526, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf527, (128, 32, 196), (6272, 1, 32), 0), buf526, reinterpret_tensor(buf521, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf522, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_240, (1536, 512), (512, 1), 0), buf787, reinterpret_tensor(primals_238, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_236, (2048, 512), (512, 1), 0), buf788, reinterpret_tensor(primals_234, (512, 512), (512, 1), 0), reinterpret_tensor(buf495, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf496, (128, 32, 196), (6272, 1, 32), 0), buf495, reinterpret_tensor(buf490, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf491, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_232, (1536, 512), (512, 1), 0), buf789, reinterpret_tensor(primals_230, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_228, (2048, 512), (512, 1), 0), buf790, reinterpret_tensor(primals_226, (512, 512), (512, 1), 0), reinterpret_tensor(buf464, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf465, (128, 32, 196), (6272, 1, 32), 0), buf464, reinterpret_tensor(buf459, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf460, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_224, (1536, 512), (512, 1), 0), buf791, reinterpret_tensor(primals_222, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_220, (2048, 512), (512, 1), 0), buf792, reinterpret_tensor(primals_218, (512, 512), (512, 1), 0), reinterpret_tensor(buf433, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf434, (128, 32, 196), (6272, 1, 32), 0), buf433, reinterpret_tensor(buf428, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf429, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_216, (1536, 512), (512, 1), 0), buf793, reinterpret_tensor(primals_214, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_212, (2048, 512), (512, 1), 0), buf794, reinterpret_tensor(primals_210, (512, 512), (512, 1), 0), reinterpret_tensor(buf402, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf403, (128, 32, 196), (6272, 1, 32), 0), buf402, reinterpret_tensor(buf397, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf398, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_208, (1536, 512), (512, 1), 0), buf795, reinterpret_tensor(primals_206, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_204, (2048, 512), (512, 1), 0), buf796, reinterpret_tensor(primals_202, (512, 512), (512, 1), 0), reinterpret_tensor(buf371, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf372, (128, 32, 196), (6272, 1, 32), 0), buf371, reinterpret_tensor(buf366, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf367, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_200, (1536, 512), (512, 1), 0), buf797, reinterpret_tensor(primals_198, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_196, (2048, 512), (512, 1), 0), buf798, reinterpret_tensor(primals_194, (512, 512), (512, 1), 0), reinterpret_tensor(buf340, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf341, (128, 32, 196), (6272, 1, 32), 0), buf340, reinterpret_tensor(buf335, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf336, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_192, (1536, 512), (512, 1), 0), buf799, reinterpret_tensor(primals_190, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_188, (2048, 512), (512, 1), 0), buf800, reinterpret_tensor(primals_186, (512, 512), (512, 1), 0), reinterpret_tensor(buf309, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf310, (128, 32, 196), (6272, 1, 32), 0), buf309, reinterpret_tensor(buf304, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf305, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_184, (1536, 512), (512, 1), 0), buf801, reinterpret_tensor(primals_182, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_180, (2048, 512), (512, 1), 0), buf802, reinterpret_tensor(primals_178, (512, 512), (512, 1), 0), reinterpret_tensor(buf278, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf279, (128, 32, 196), (6272, 1, 32), 0), buf278, reinterpret_tensor(buf273, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf274, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_176, (1536, 512), (512, 1), 0), buf803, reinterpret_tensor(primals_174, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_172, (2048, 512), (512, 1), 0), buf804, reinterpret_tensor(primals_170, (512, 512), (512, 1), 0), reinterpret_tensor(buf247, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf248, (128, 32, 196), (6272, 1, 32), 0), buf247, reinterpret_tensor(buf242, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf243, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_168, (1536, 512), (512, 1), 0), buf805, reinterpret_tensor(primals_166, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_164, (2048, 512), (512, 1), 0), buf806, reinterpret_tensor(primals_162, (512, 512), (512, 1), 0), reinterpret_tensor(buf216, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf217, (128, 32, 196), (6272, 1, 32), 0), buf216, reinterpret_tensor(buf211, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf212, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_160, (1536, 512), (512, 1), 0), buf807, reinterpret_tensor(primals_158, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_156, (2048, 512), (512, 1), 0), buf808, reinterpret_tensor(primals_154, (512, 512), (512, 1), 0), reinterpret_tensor(buf185, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf186, (128, 32, 196), (6272, 1, 32), 0), buf185, reinterpret_tensor(buf180, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf181, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_152, (1536, 512), (512, 1), 0), buf809, reinterpret_tensor(primals_150, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_148, (2048, 512), (512, 1), 0), buf810, reinterpret_tensor(primals_146, (512, 512), (512, 1), 0), reinterpret_tensor(buf154, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf155, (128, 32, 196), (6272, 1, 32), 0), buf154, reinterpret_tensor(buf149, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf150, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_144, (1536, 512), (512, 1), 0), buf811, buf812, reinterpret_tensor(primals_140, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_138, (1024, 256), (256, 1), 0), buf813, reinterpret_tensor(primals_136, (256, 256), (256, 1), 0), reinterpret_tensor(buf110, (256, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf111, (256, 32, 196), (6272, 1, 32), 0), buf110, reinterpret_tensor(buf105, (256, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf106, (256, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_134, (768, 256), (256, 1), 0), buf814, reinterpret_tensor(primals_132, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_130, (1024, 256), (256, 1), 0), buf815, reinterpret_tensor(primals_128, (256, 256), (256, 1), 0), reinterpret_tensor(buf79, (256, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf80, (256, 32, 196), (6272, 1, 32), 0), buf79, reinterpret_tensor(buf74, (256, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf75, (256, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_126, (768, 256), (256, 1), 0), buf816, buf817, reinterpret_tensor(primals_122, (128, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 128), (128, 1), 0), buf818, reinterpret_tensor(primals_118, (128, 128), (128, 1), 0), reinterpret_tensor(buf37, (512, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf38, (512, 32, 196), (6272, 1, 32), 0), buf37, reinterpret_tensor(buf32, (512, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf33, (512, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_116, (384, 128), (128, 1), 0), buf819, reinterpret_tensor(primals_114, (128, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 128), (128, 1), 0), buf820, reinterpret_tensor(primals_110, (128, 128), (128, 1), 0), reinterpret_tensor(buf12, (512, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf13, (512, 32, 196), (6272, 1, 32), 0), buf12, reinterpret_tensor(buf7, (512, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf8, (512, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_108, (384, 128), (128, 1), 0), buf821, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
