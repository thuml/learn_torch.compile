
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


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2pireug6rdmcbcakb5biu7ggk5jqzu4blznrmd5fad7q2edxnk.py
# Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2, l__mod___patch_embed_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___patch_embed_conv_1 => add_1, mul_1, mul_2, sub
# l__mod___patch_embed_conv_2 => relu
# l__mod___patch_embed_conv_3 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/lj/cljkq2kumh5kd4czm7bismkglhp4finq34bvznzeup3wdulvrq2k.py
# Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm1 => clone, var_mean
triton_red_fused_native_layer_norm_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (75264*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jisnvq4flxipjdzaddh5dywcpjnp37glmreep2rk5zx4qcoqpr.py
# Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm1 => clone, var_mean
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': []}
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
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (1568*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mbyr5cruiu7nveb6redykteqypzdq42zy5e6yrpmddii6lyrxm.py
# Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm1 => add_6, add_7, clone, mul_10, mul_9, rsqrt, sub_3, var_mean
triton_poi_fused_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_3', 'mutated_arg_names': []},
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
    x0 = xindex % 784
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 192.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwbst3vizthlbu7ollpudjyobmpihapktqra54cogon5oq5cwmad.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___network_0___0___attn_pool => avg_pool2d
triton_poi_fused_avg_pool2d_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxwfffhiagl6rna7rveajfzwmaqjev7cjsgtfpif3exygzobw7z.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (37632*(x0 // 196)) + (x0 % 196)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzt546gbtlmzs6ah5pnd3xbi6cw5duqgct5mntz3zc36ijzg66w.py
# Source Nodes: [attn_2, attn_3], Original ATen: [aten._softmax, aten.mul]
# attn_2 => mul_11
# attn_3 => amax, clone_2, div, exp, sub_4, sum_1
triton_per_fused__softmax_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 54
    x4 = xindex % 9
    x5 = (xindex // 9) % 6
    x6 = (xindex // 54) % 196
    x7 = (xindex // 10584)
    tmp0 = tl.load(in_ptr0 + (r2 + (9*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (9*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (9*x4) + (81*x6) + (15876*x5) + (95256*x7)), tmp15, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pwgvn5ryzzphktbyrev3pvx7ujemqyuhuyqpjd45qud3pedtnq.py
# Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
# getattr_l__mod___network_0___0___attn_v => mm
triton_poi_fused_mm_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (150528*(x0 // 784)) + (x0 % 784)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hyjep7yjy4zzrors2w5cefktxgsu4yoij6h7hwglsd3wdzv2k5.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_4
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = (-1) + (2*(x2 // 14)) + (x1 // 3) + (tl.where(((2*(x2 // 14)) + (x1 // 3)) >= 0, 0, 30))
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*(x2 % 14)) + (x1 % 3) + (tl.where(((2*(x2 % 14)) + (x1 % 3)) >= 0, 0, 30))
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5568) + x0 + (32*x3) + (192*(x1 % 3)) + (192*(tl.where(((2*(x2 % 14)) + (x1 % 3)) >= 0, 0, 30))) + (384*(x2 % 14)) + (5376*(x1 // 3)) + (5376*(tl.where(((2*(x2 // 14)) + (x1 // 3)) >= 0, 0, 30))) + (10752*(x2 // 14)) + (150528*x4)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x6), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bfsuqz2luxukyffkcgafkcbs4c2bnzudeucqmuylwl6dbb5kar.py
# Source Nodes: [x_4], Original ATen: [aten.col2im]
# x_4 => full_default
triton_poi_fused_col2im_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1382400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c447kjwrpl4arrdhntdmd3xdq7dhjaz5faw3jwso4tndcvnbbpuw.py
# Source Nodes: [x_3, x_4], Original ATen: [aten.clone, aten.col2im]
# x_3 => clone_5
# x_4 => _unsafe_index_put, full_default
triton_poi_fused_clone_col2im_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_col2im_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 196
    x3 = (xindex // 196)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (288*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1764*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7ncga3sb5ouvz5xz7jpbqnmeibu3rrlidx5wzpshdase3s6wzr.py
# Source Nodes: [x_4], Original ATen: [aten.col2im]
# x_4 => add_10
triton_poi_fused_col2im_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = x1 + (2*x0)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3hoj4xeetjfezb45zoiwvzswyifp64wwbs77fvv7fbo6uctlwt.py
# Source Nodes: [x_5], Original ATen: [aten.clone]
# x_5 => clone_6
triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 28) % 28
    y0 = yindex % 28
    x3 = xindex
    y2 = (yindex // 784)
    y5 = yindex
    tmp0 = 1 + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + y0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (31 + y0 + (30*y1) + (900*x3) + (172800*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x3 + (192*y5)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqnbflg5r3jqjegvx5dfhrxkbabm6of2jjlhavpjojcfa5blpqpd.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm2 => clone_8, var_mean_1
# x_5 => add_12
# x_7 => add_13
triton_red_fused_add_native_layer_norm_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 784
    x2 = (xindex // 1568)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (75264*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (96*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ye4p4umjzlu6wul2sdvshu5iazzbltmgb3dtwib3alv65wftil.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm2 => clone_8, var_mean_1
# x_5 => add_12
# x_7 => add_13
triton_per_fused_add_native_layer_norm_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/44/c44xe7uap3t7ywkxp4wsebdm2ncvtd6ig3dhngdmbtj26qah3qv5.py
# Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___0___norm2 => add_14, add_15, clone_8, mul_12, mul_13, rsqrt_1, sub_5, var_mean_1
# x_5 => add_12
# x_7 => add_13
triton_poi_fused_add_native_layer_norm_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 192.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltuduxjtt2pbg3xh3alymx2vd25rvkolbnnwbabghoz4wsrzjaq.py
# Source Nodes: [x_9], Original ATen: [aten.gelu]
# x_9 => add_16, erf, mul_14, mul_15, mul_16
triton_poi_fused_gelu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 576
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tqs67bofotsoktzlivquwtu6vj42tvasotca6zh6uhtoa7exue.py
# Source Nodes: [getattr_l__mod___network_0___1___norm1, x_14, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___1___norm1 => add_18, add_19, clone_11, mul_17, mul_18, rsqrt_2, sub_6, var_mean_2
# x_14 => add_17
# x_5 => add_12
# x_7 => add_13
triton_per_fused_add_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (150528*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = 192.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp10, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (192*x3)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bs2yjdx6c2yhiulw4724dbhgy4kejjee3neyji27fjyemyf7jk.py
# Source Nodes: [getattr_l__mod___network_0___1___attn_pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___network_0___1___attn_pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y0 = yindex % 192
    y1 = (yindex // 192)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (10752*x3) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (192 + y0 + (384*x2) + (10752*x3) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (5376 + y0 + (384*x2) + (10752*x3) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5568 + y0 + (384*x2) + (10752*x3) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x5 + (196*y4)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2f5kzyh3pcfdirm27zoiff32tu5kkdw6tngdgzpdq34xpsihabe.py
# Source Nodes: [getattr_l__mod___network_0___1___norm2, x_17, x_19], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___1___norm2 => add_26, add_27, clone_19, mul_20, mul_21, rsqrt_3, sub_8, var_mean_3
# x_17 => add_24
# x_19 => add_25
triton_per_fused_add_native_layer_norm_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6tlmhdjigxmm2m76nnvk3lxun7asce3ryqcw7wzvtix4nd3s5i.py
# Source Nodes: [getattr_l__mod___network_0___2___norm1, x_17, x_19, x_26], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_0___2___norm1 => add_30, add_31, clone_22, mul_25, mul_26, rsqrt_4, sub_9, var_mean_4
# x_17 => add_24
# x_19 => add_25
# x_26 => add_29
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cibsxryxe6nappmcgtqshgak7fmllc5bxsynxgt3zsaevqehnuoh.py
# Source Nodes: [x_53], Original ATen: [aten.convolution]
# x_53 => convolution_4
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q7/cq7jnii36j26sffnntdkrbngtbju4buo2uraawzrccj55nzeoeic.py
# Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___0___norm1 => clone_45, var_mean_8
# x_56 => add_54
triton_red_fused_add_native_layer_norm_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x5 = xindex % 588
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (75264*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x6), tmp6, xmask)
    tl.store(out_ptr1 + (x6), tmp7, xmask)
    tl.store(out_ptr2 + (x6), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgsw6arzzdvrb7zzvq5me2x7h3g6df5fptudljhyr6kbwlicp6g6.py
# Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___0___norm1 => clone_45, var_mean_8
# x_56 => add_54
triton_per_fused_add_native_layer_norm_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tk/ctkarpd2jkcttnxpseipzxyuyyjihf2vc6qepanbq5sdikg7pifu.py
# Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___0___norm1 => add_55, add_56, clone_45, mul_41, mul_42, rsqrt_8, sub_15, var_mean_8
# x_56 => add_54
triton_poi_fused_add_native_layer_norm_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qzsahyybaponojtpx2u4v27vguugni4pc4i7kprfx6ipr6vtes.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_46
triton_poi_fused_clone_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 12
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwq7wugbcm46d22chrdx3ofzvpumrmacaplhwa2f5g6uufknl4i.py
# Source Nodes: [matmul_4], Original ATen: [aten.clone]
# matmul_4 => clone_47
triton_poi_fused_clone_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (1152*x2) + (225792*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6s7okhfvhwpde2vcefpuaafwdwmfqanhawfvw5grf7djx2lpjoz.py
# Source Nodes: [attn_20, attn_21], Original ATen: [aten._softmax, aten.mul]
# attn_20 => mul_43
# attn_21 => amax_4, div_4, exp_4, sub_16, sum_5
triton_per_fused__softmax_mul_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18816
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
    tmp1 = 0.1767766952966369
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
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kv2a7ercz27h4ywdh7nljapvshb4rwiwr6sgwoye7ncq6yzpag.py
# Source Nodes: [matmul_5], Original ATen: [aten.clone]
# matmul_5 => clone_49
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 12
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (32*x2) + (1152*x1) + (225792*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf376abb6boljhupb4s5n2itsyyedibasj44qnj3gmsc6qrlddta.py
# Source Nodes: [x_58], Original ATen: [aten.clone]
# x_58 => clone_50
triton_poi_fused_clone_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 12
    x2 = (xindex // 384) % 196
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (6272*x1) + (75264*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csm7keuvbykeawc5q4hiqvva5osz5abneiizu732t5s6zgy7u3dg.py
# Source Nodes: [getattr_l__mod___network_2___0___norm2, x_56, x_61], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___0___norm2 => add_58, add_59, clone_52, mul_44, mul_45, rsqrt_9, sub_17, var_mean_9
# x_56 => add_54
# x_61 => add_57
triton_per_fused_add_native_layer_norm_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/crurvliesrstimnzpxhlgbusfpwdqdf4zdzxme6d6nahgyv6c6xc.py
# Source Nodes: [x_63], Original ATen: [aten.gelu]
# x_63 => add_60, erf_4, mul_46, mul_47, mul_48
triton_poi_fused_gelu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckoeejprcs2o2glv2c57cr4bcsjl5kyxsl5waq6jjdue3hhsfasx.py
# Source Nodes: [getattr_l__mod___network_2___1___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___1___norm1 => add_62, add_63, clone_55, mul_49, mul_50, rsqrt_10, sub_18, var_mean_10
# x_68 => add_61
triton_per_fused_add_native_layer_norm_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63ngtj34jp5qi7mvfcc75nnnhqr7kw5gphkbd6wsz3ewumarg2e.py
# Source Nodes: [getattr_l__mod___network_2___1___norm2, x_68, x_72], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_l__mod___network_2___1___norm2 => add_65, add_66, clone_62, mul_52, mul_53, rsqrt_11, sub_20, var_mean_11
# x_68 => add_61
# x_72 => add_64
triton_per_fused_add_native_layer_norm_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_33', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
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
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtxp57ljh3kk6lc2x3z5moxsltvti2wlhy5vbyuootxvzn23zue.py
# Source Nodes: [cat_5, l__mod___post_network_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_5 => cat
# l__mod___post_network_0_norm1 => var_mean_36
triton_per_fused_cat_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 384, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmj5kykaa6mfltedml5t6kkijycb3ajr6xrszhllatrkucbs2i6.py
# Source Nodes: [cat_5, l__mod___post_network_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_5 => cat
# l__mod___post_network_0_norm1 => add_153, add_154, mul_153, mul_154, rsqrt_36, sub_57, var_mean_36
triton_poi_fused_cat_native_layer_norm_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp19 = tl.load(in_ptr4 + (x2 + (197*y1)), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2 + (197*y1)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 384.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (384*x2) + (75648*y1)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pw/cpwuwriqizdwraib45c7hcyeszfybmv3ad6fmkenor3uhhvfvdbg.py
# Source Nodes: [mul_18], Original ATen: [aten.mul]
# mul_18 => mul_155
triton_poi_fused_mul_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cnfpvjv2hgwsimhr64rlep4txgwy7b3e2noxa4xbok3tg73jb4.py
# Source Nodes: [attn_62], Original ATen: [aten.clone]
# attn_62 => clone_185
triton_poi_fused_clone_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
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
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (151296*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckizv7jsm76y72zllq5yv4c5plmftbtjolhnzml37do64x7r5qgl.py
# Source Nodes: [attn_63], Original ATen: [aten._softmax]
# attn_63 => amax_18, div_18, exp_18, sub_58, sum_19
triton_per_fused__softmax_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (197*x0)), tmp11, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2n7kk6xb7wh5etrp6marhlva5goz2os4wxewalytrbrucj36du.py
# Source Nodes: [matmul_33], Original ATen: [aten.clone]
# matmul_33 => clone_187
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 197
    x2 = (xindex // 6304) % 12
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (32*x2) + (768*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwljlvnwyvv6pnojk3kvwqafwcpjevmzleu4yh3qrfy7yb4tgzhv.py
# Source Nodes: [cls_embed_4, l__mod___post_network_0_norm2], Original ATen: [aten.add, aten.native_layer_norm]
# cls_embed_4 => add_155
# l__mod___post_network_0_norm2 => add_156, add_157, mul_156, mul_157, rsqrt_37, sub_59, var_mean_37
triton_per_fused_add_native_layer_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp18 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1], 197, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (74880 + r1 + (75264*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (74880 + r1 + (75264*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tl.full([1], 384, tl.int32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp21 - tmp31
    tmp39 = 384.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-05
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp48, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c532hx3daizwbduext42q4ptzvuaruspjng7u3p6d5y6qnhwvfii.py
# Source Nodes: [x_219], Original ATen: [aten.gelu]
# x_219 => add_158, erf_18, mul_158, mul_159, mul_160
triton_poi_fused_gelu_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/cod6ksjepxxlimv3hwj5i2jgitdboavofokmmvrlviudfxfbps2w.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_1
triton_poi_fused_cat_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tmp17 & tmp12
    tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp15 >= tmp3
    tmp23 = tmp15 < tmp13
    tmp24 = tmp22 & tmp12
    tmp25 = tl.load(in_ptr4 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr5 + (y0 + (384*(((-1) + x2) % 196)) + (75264*y1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr6 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp24, tmp29, tmp30)
    tmp32 = tl.where(tmp17, tmp21, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp12, tmp32, tmp33)
    tmp35 = tl.where(tmp4, tmp11, tmp34)
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f4/cf4ufwcfeslt2ayzq3vicn64xyehug2hya3vnrq5qzufvoafoyl6.py
# Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___post_network_1_norm1 => var_mean_38
triton_red_fused_native_layer_norm_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4728
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 197
    x2 = (xindex // 591)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (197*r3) + (25216*x0) + (75648*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x1 + (197*x0) + (591*x2)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (197*x0) + (591*x2)), tmp3, xmask)
    tl.store(out_ptr2 + (x1 + (197*x0) + (591*x2)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5fbtlqht3vdyv3dbygg3bkizrbqxlyullubiljnw6lcwxdhz7t.py
# Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___post_network_1_norm1 => var_mean_38
triton_per_fused_native_layer_norm_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ev/cev4fmsnfjy55zyvtntoqnrdagjpitscj7qinorwxkaljcmw72zw.py
# Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
# l__mod___post_network_1_norm1 => add_160, add_161, mul_161, mul_162, rsqrt_38, sub_60, var_mean_38
triton_poi_fused_native_layer_norm_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 197
    y1 = (yindex // 197)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (197*x2) + (75648*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cna5kd2s5d4iavxn3kqxkf5qz6h4tczmv3ikyzsygqrhhqszofhv.py
# Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
# cls_embed_10 => add_162
# l__mod___post_network_1_norm2 => var_mean_39
triton_red_fused_add_native_layer_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 3
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((197*r2) + (25216*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clis6iyuh67zsagbwgdf5jznwuv3oktjf43rc5e4gubjnwoweg4r.py
# Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
# cls_embed_10 => add_162
# l__mod___post_network_1_norm2 => var_mean_39
triton_per_fused_add_native_layer_norm_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rm/crmoxgubkz5qqfyau3ndw4aloeomlirkrtsp7xdfwytedu7i5r3e.py
# Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
# cls_embed_10 => add_162
# l__mod___post_network_1_norm2 => add_163, add_164, mul_164, mul_165, rsqrt_39, sub_62, var_mean_39
triton_poi_fused_add_native_layer_norm_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + (197*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jitimsztkpp2d4rj6mf2kglyfm76z6tr7qbeuouhexvnum2zdr.py
# Source Nodes: [cat_3], Original ATen: [aten.cat]
# cat_3 => cat_2
triton_poi_fused_cat_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 197
    x2 = xindex
    y1 = (yindex // 197)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((197*x2) + (75648*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2 + (384*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(x2, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.load(in_ptr3 + (x2 + (384*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x2, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 197, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (y0 + (197*x2) + (75648*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tl.store(out_ptr0 + (y0 + (197*x2) + (75648*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5nyobpsilx3rnpfsv4zcabvfc6k5cqn3ax33qzzmqh67iqjkfg.py
# Source Nodes: [x_234], Original ATen: [aten.native_layer_norm]
# x_234 => var_mean_40
triton_red_fused_native_layer_norm_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4728
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (197*r2) + (25216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2g4csgofndrrguvndpdbduuntrkrtsbzgfocxw5vebenxheil5.py
# Source Nodes: [x_234], Original ATen: [aten.native_layer_norm]
# x_234 => add_167, add_168, mul_169, mul_170, rsqrt_40, sub_63, var_mean_40
triton_poi_fused_native_layer_norm_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 197
    x2 = (xindex // 75648)
    x1 = (xindex // 197) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (197*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 + (197*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbuaddy3vecajt6syr5eeszivkcrpyd3r6ybyryouhoykje3r7m.py
# Source Nodes: [aux], Original ATen: [aten.clone]
# aux => clone_198
triton_poi_fused_clone_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (1 + y0 + (197*x2) + (75648*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirzbbbrjzks7w6f45pujlpubkysuwhd36aaqmpuc2g2qb6mw7eu.py
# Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
# aux => add_169
# max_1 => max_1
triton_red_fused_add_max_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16000
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (98000*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz7t5saqpfwdnzr2i5lh5gnczezqls3wwwdo7u3b6qw35t3g5xg.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (197*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czbdrejp6fxftbfmqn23rj63ybxgdnhlaqe3xpt7cxfi6nm7osp4.py
# Source Nodes: [aux, max_1, mul_20, x_236], Original ATen: [aten.add, aten.max, aten.mul]
# aux => add_169
# max_1 => max_1
# mul_20 => mul_171
# x_236 => add_170
triton_per_fused_add_max_mul_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_mul_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (2000*x1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp7 = tmp5 + tmp6
    tmp8 = 0.5
    tmp9 = tmp4 * tmp8
    tmp10 = tmp7 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, 192), (192, 1))
    assert_size_stride(arg16_1, (486, 192), (192, 1))
    assert_size_stride(arg17_1, (486, ), (1, ))
    assert_size_stride(arg18_1, (192, 192), (192, 1))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (576, 192), (192, 1))
    assert_size_stride(arg23_1, (576, ), (1, ))
    assert_size_stride(arg24_1, (192, 576), (576, 1))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, 192), (192, 1))
    assert_size_stride(arg29_1, (486, 192), (192, 1))
    assert_size_stride(arg30_1, (486, ), (1, ))
    assert_size_stride(arg31_1, (192, 192), (192, 1))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (576, 192), (192, 1))
    assert_size_stride(arg36_1, (576, ), (1, ))
    assert_size_stride(arg37_1, (192, 576), (576, 1))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, 192), (192, 1))
    assert_size_stride(arg42_1, (486, 192), (192, 1))
    assert_size_stride(arg43_1, (486, ), (1, ))
    assert_size_stride(arg44_1, (192, 192), (192, 1))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (576, 192), (192, 1))
    assert_size_stride(arg49_1, (576, ), (1, ))
    assert_size_stride(arg50_1, (192, 576), (576, 1))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (192, 192), (192, 1))
    assert_size_stride(arg55_1, (486, 192), (192, 1))
    assert_size_stride(arg56_1, (486, ), (1, ))
    assert_size_stride(arg57_1, (192, 192), (192, 1))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (576, 192), (192, 1))
    assert_size_stride(arg62_1, (576, ), (1, ))
    assert_size_stride(arg63_1, (192, 576), (576, 1))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (1152, 384), (384, 1))
    assert_size_stride(arg70_1, (384, 384), (384, 1))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (1152, 384), (384, 1))
    assert_size_stride(arg75_1, (1152, ), (1, ))
    assert_size_stride(arg76_1, (384, 1152), (1152, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (1152, 384), (384, 1))
    assert_size_stride(arg81_1, (384, 384), (384, 1))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (1152, 384), (384, 1))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (384, 1152), (1152, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1152, 384), (384, 1))
    assert_size_stride(arg92_1, (384, 384), (384, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (1152, 384), (384, 1))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (384, 1152), (1152, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (1152, 384), (384, 1))
    assert_size_stride(arg103_1, (384, 384), (384, 1))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (1152, 384), (384, 1))
    assert_size_stride(arg108_1, (1152, ), (1, ))
    assert_size_stride(arg109_1, (384, 1152), (1152, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (1152, 384), (384, 1))
    assert_size_stride(arg114_1, (384, 384), (384, 1))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (1152, 384), (384, 1))
    assert_size_stride(arg119_1, (1152, ), (1, ))
    assert_size_stride(arg120_1, (384, 1152), (1152, 1))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (1152, 384), (384, 1))
    assert_size_stride(arg125_1, (384, 384), (384, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (1152, 384), (384, 1))
    assert_size_stride(arg130_1, (1152, ), (1, ))
    assert_size_stride(arg131_1, (384, 1152), (1152, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (1152, 384), (384, 1))
    assert_size_stride(arg136_1, (384, 384), (384, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (1152, 384), (384, 1))
    assert_size_stride(arg141_1, (1152, ), (1, ))
    assert_size_stride(arg142_1, (384, 1152), (1152, 1))
    assert_size_stride(arg143_1, (384, ), (1, ))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (1152, 384), (384, 1))
    assert_size_stride(arg147_1, (384, 384), (384, 1))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (1152, 384), (384, 1))
    assert_size_stride(arg152_1, (1152, ), (1, ))
    assert_size_stride(arg153_1, (384, 1152), (1152, 1))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (1152, 384), (384, 1))
    assert_size_stride(arg158_1, (384, 384), (384, 1))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (1152, 384), (384, 1))
    assert_size_stride(arg163_1, (1152, ), (1, ))
    assert_size_stride(arg164_1, (384, 1152), (1152, 1))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (1152, 384), (384, 1))
    assert_size_stride(arg169_1, (384, 384), (384, 1))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (1152, 384), (384, 1))
    assert_size_stride(arg174_1, (1152, ), (1, ))
    assert_size_stride(arg175_1, (384, 1152), (1152, 1))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (1152, 384), (384, 1))
    assert_size_stride(arg180_1, (384, 384), (384, 1))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (1152, 384), (384, 1))
    assert_size_stride(arg185_1, (1152, ), (1, ))
    assert_size_stride(arg186_1, (384, 1152), (1152, 1))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (1152, 384), (384, 1))
    assert_size_stride(arg191_1, (384, 384), (384, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (1152, 384), (384, 1))
    assert_size_stride(arg196_1, (1152, ), (1, ))
    assert_size_stride(arg197_1, (384, 1152), (1152, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (1152, 384), (384, 1))
    assert_size_stride(arg202_1, (384, 384), (384, 1))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (1152, 384), (384, 1))
    assert_size_stride(arg207_1, (1152, ), (1, ))
    assert_size_stride(arg208_1, (384, 1152), (1152, 1))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (1152, 384), (384, 1))
    assert_size_stride(arg213_1, (384, 384), (384, 1))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (1152, 384), (384, 1))
    assert_size_stride(arg218_1, (1152, ), (1, ))
    assert_size_stride(arg219_1, (384, 1152), (1152, 1))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (768, 384), (384, 1))
    assert_size_stride(arg224_1, (384, 384), (384, 1))
    assert_size_stride(arg225_1, (384, 384), (384, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, ), (1, ))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (1152, 384), (384, 1))
    assert_size_stride(arg230_1, (1152, ), (1, ))
    assert_size_stride(arg231_1, (384, 1152), (1152, 1))
    assert_size_stride(arg232_1, (384, ), (1, ))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (768, 384), (384, 1))
    assert_size_stride(arg236_1, (384, 384), (384, 1))
    assert_size_stride(arg237_1, (384, 384), (384, 1))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (1152, 384), (384, 1))
    assert_size_stride(arg242_1, (1152, ), (1, ))
    assert_size_stride(arg243_1, (384, 1152), (1152, 1))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (1000, 384), (384, 1))
    assert_size_stride(arg248_1, (1000, ), (1, ))
    assert_size_stride(arg249_1, (1000, 384), (384, 1))
    assert_size_stride(arg250_1, (1000, ), (1, ))
    assert_size_stride(arg251_1, (64, ), (1, ))
    assert_size_stride(arg252_1, (64, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (64, ), (1, ))
    assert_size_stride(arg255_1, (64, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___patch_embed_conv_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg260_1, arg2_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg260_1
        del arg2_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2, l__mod___patch_embed_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg251_1, arg252_1, arg3_1, arg4_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg251_1
        del arg252_1
        del arg3_1
        del arg4_1
        # Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2, l__mod___patch_embed_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg5_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___patch_embed_conv_4, l__mod___patch_embed_conv_5, l__mod___patch_embed_conv_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf3, arg254_1, arg255_1, arg6_1, arg7_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg254_1
        del arg255_1
        del arg6_1
        del arg7_1
        # Source Nodes: [l__mod___patch_embed_conv_4, l__mod___patch_embed_conv_5, l__mod___patch_embed_conv_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg8_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg8_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [l__mod___patch_embed_conv_7, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf5, arg257_1, arg258_1, arg9_1, arg10_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg10_1
        del arg257_1
        del arg258_1
        del arg9_1
        # Source Nodes: [l__mod___patch_embed_conv_7, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg11_1
        del buf5
        buf7 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf8 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_1.run(buf6, arg12_1, buf7, buf8, buf9, 12544, 96, grid=grid(12544), stream=stream0)
        buf10 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        buf11 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf7, buf8, buf9, buf10, buf11, 6272, 2, grid=grid(6272), stream=stream0)
        buf13 = empty_strided((8, 28, 28, 192), (150528, 28, 1, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_3.run(buf6, arg12_1, buf10, buf11, arg13_1, arg14_1, buf13, 1204224, grid=grid(1204224), stream=stream0)
        del arg13_1
        del arg14_1
        buf14 = empty((8, 192, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_4.run(buf13, buf14, 301056, grid=grid(301056), stream=stream0)
        buf15 = empty_strided((1568, 192), (1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf14, buf15, 301056, grid=grid(301056), stream=stream0)
        buf16 = empty((1568, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf15, reinterpret_tensor(arg16_1, (192, 486), (1, 192), 0), out=buf16)
        del arg16_1
        buf21 = empty((8, 6, 196, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, attn_3], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_6.run(buf16, arg17_1, buf21, 84672, 9, grid=grid(84672), stream=stream0)
        del arg17_1
        buf19 = empty_strided((6272, 192), (1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
        triton_poi_fused_mm_7.run(buf13, buf19, 1204224, grid=grid(1204224), stream=stream0)
        buf20 = reinterpret_tensor(buf13, (6272, 192), (192, 1), 0); del buf13  # reuse
        # Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(arg15_1, (192, 192), (1, 192), 0), out=buf20)
        del arg15_1
        buf22 = empty((8, 6, 196, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf20, buf22, 2709504, grid=grid(2709504), stream=stream0)
        buf23 = empty((9408, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf22, (9408, 9, 32), (288, 32, 1), 0), out=buf23)
        buf24 = empty((8, 192, 30, 30), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_9.run(buf24, 1382400, grid=grid(1382400), stream=stream0)
        buf25 = reinterpret_tensor(buf22, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf22  # reuse
        buf26 = reinterpret_tensor(buf25, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf25  # reuse
        # Source Nodes: [x_3, x_4], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_10.run(buf26, buf23, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf27 = empty((3, 14), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf27, 42, grid=grid(42), stream=stream0)
        buf28 = empty((3, 14), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_4], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf28, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf24, [None, None, reinterpret_tensor(buf27, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf28], buf26, True)
        buf31 = reinterpret_tensor(buf20, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf20  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf24, buf31, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf32 = reinterpret_tensor(buf19, (6272, 192), (192, 1), 0); del buf19  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (6272, 192), (192, 1), 0), reinterpret_tensor(arg18_1, (192, 192), (1, 192), 0), out=buf32)
        del arg18_1
        buf33 = reinterpret_tensor(buf9, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf9  # reuse
        buf34 = reinterpret_tensor(buf8, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf8  # reuse
        buf35 = reinterpret_tensor(buf7, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf7  # reuse
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_13.run(buf6, arg12_1, buf32, arg19_1, buf33, buf34, buf35, 12544, 96, grid=grid(12544), stream=stream0)
        buf36 = buf11; del buf11  # reuse
        buf37 = buf10; del buf10  # reuse
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_14.run(buf33, buf34, buf35, buf36, buf37, 6272, 2, grid=grid(6272), stream=stream0)
        del buf33
        del buf34
        del buf35
        buf39 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_l__mod___network_0___0___norm2, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_15.run(buf6, arg12_1, buf32, arg19_1, buf36, buf37, arg20_1, arg21_1, buf39, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg20_1
        del arg21_1
        del buf36
        del buf37
        buf40 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (6272, 192), (192, 1), 0), reinterpret_tensor(arg22_1, (192, 576), (1, 192), 0), out=buf40)
        del arg22_1
        buf41 = reinterpret_tensor(buf40, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf40  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf41, arg23_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg23_1
        buf42 = reinterpret_tensor(buf39, (6272, 192), (192, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf41, (6272, 576), (576, 1), 0), reinterpret_tensor(arg24_1, (576, 192), (1, 576), 0), out=buf42)
        del arg24_1
        buf43 = reinterpret_tensor(buf32, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf32  # reuse
        buf47 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_0___1___norm1, x_14, x_5, x_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf43, buf6, arg12_1, arg19_1, buf42, arg25_1, arg26_1, arg27_1, buf47, 6272, 192, grid=grid(6272), stream=stream0)
        del arg12_1
        del arg19_1
        del arg25_1
        del arg26_1
        del arg27_1
        buf48 = reinterpret_tensor(buf15, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf15  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___attn_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_18.run(buf47, buf48, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf49 = reinterpret_tensor(buf14, (1568, 192), (1, 1568), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf48, buf49, 301056, grid=grid(301056), stream=stream0)
        buf50 = reinterpret_tensor(buf21, (1568, 486), (486, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf49, reinterpret_tensor(arg29_1, (192, 486), (1, 192), 0), out=buf50)
        del arg29_1
        buf54 = reinterpret_tensor(buf16, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf16  # reuse
        # Source Nodes: [attn_7, attn_8], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_6.run(buf50, arg30_1, buf54, 84672, 9, grid=grid(84672), stream=stream0)
        del arg30_1
        buf53 = reinterpret_tensor(buf6, (6272, 192), (192, 1), 0); del buf6  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (6272, 192), (192, 1), 0), reinterpret_tensor(arg28_1, (192, 192), (1, 192), 0), out=buf53)
        del arg28_1
        buf55 = reinterpret_tensor(buf26, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf26  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf53, buf55, 2709504, grid=grid(2709504), stream=stream0)
        buf56 = buf23; del buf23  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf55, (9408, 9, 32), (288, 32, 1), 0), out=buf56)
        buf57 = buf24; del buf24  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_9.run(buf57, 1382400, grid=grid(1382400), stream=stream0)
        buf58 = reinterpret_tensor(buf55, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf55  # reuse
        buf59 = reinterpret_tensor(buf58, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf58  # reuse
        # Source Nodes: [x_15, x_16], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_10.run(buf59, buf56, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf60 = buf28; del buf28  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf60, 42, grid=grid(42), stream=stream0)
        buf61 = buf27; del buf27  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf61, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf57, [None, None, reinterpret_tensor(buf60, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf61], buf59, True)
        buf64 = reinterpret_tensor(buf53, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf53  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf57, buf64, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf65 = reinterpret_tensor(buf47, (6272, 192), (192, 1), 0); del buf47  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (6272, 192), (192, 1), 0), reinterpret_tensor(arg31_1, (192, 192), (1, 192), 0), out=buf65)
        del arg31_1
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_l__mod___network_0___1___norm2, x_17, x_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf43, buf65, arg32_1, arg33_1, arg34_1, buf69, 6272, 192, grid=grid(6272), stream=stream0)
        del arg33_1
        del arg34_1
        buf70 = reinterpret_tensor(buf41, (6272, 576), (576, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (6272, 192), (192, 1), 0), reinterpret_tensor(arg35_1, (192, 576), (1, 192), 0), out=buf70)
        del arg35_1
        buf71 = reinterpret_tensor(buf70, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf70  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf71, arg36_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg36_1
        buf72 = reinterpret_tensor(buf69, (6272, 192), (192, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (6272, 576), (576, 1), 0), reinterpret_tensor(arg37_1, (576, 192), (1, 576), 0), out=buf72)
        del arg37_1
        buf73 = reinterpret_tensor(buf72, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf72  # reuse
        buf77 = reinterpret_tensor(buf42, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf42  # reuse
        # Source Nodes: [getattr_l__mod___network_0___2___norm1, x_17, x_19, x_26], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf73, buf43, buf65, arg32_1, arg38_1, arg39_1, arg40_1, buf77, 6272, 192, grid=grid(6272), stream=stream0)
        del arg32_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf78 = reinterpret_tensor(buf49, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf49  # reuse
        # Source Nodes: [getattr_l__mod___network_0___2___attn_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_18.run(buf77, buf78, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf79 = reinterpret_tensor(buf48, (1568, 192), (1, 1568), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf78, buf79, 301056, grid=grid(301056), stream=stream0)
        buf80 = reinterpret_tensor(buf54, (1568, 486), (486, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf79, reinterpret_tensor(arg42_1, (192, 486), (1, 192), 0), out=buf80)
        del arg42_1
        buf84 = reinterpret_tensor(buf50, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf50  # reuse
        # Source Nodes: [attn_12, attn_13], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_6.run(buf80, arg43_1, buf84, 84672, 9, grid=grid(84672), stream=stream0)
        del arg43_1
        buf83 = buf65; del buf65  # reuse
        # Source Nodes: [getattr_l__mod___network_0___2___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (6272, 192), (192, 1), 0), reinterpret_tensor(arg41_1, (192, 192), (1, 192), 0), out=buf83)
        del arg41_1
        buf85 = reinterpret_tensor(buf59, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf59  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf83, buf85, 2709504, grid=grid(2709504), stream=stream0)
        buf86 = buf56; del buf56  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf85, (9408, 9, 32), (288, 32, 1), 0), out=buf86)
        buf87 = buf57; del buf57  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_9.run(buf87, 1382400, grid=grid(1382400), stream=stream0)
        buf88 = reinterpret_tensor(buf85, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf85  # reuse
        buf89 = reinterpret_tensor(buf88, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf88  # reuse
        # Source Nodes: [x_27, x_28], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_10.run(buf89, buf86, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf90 = buf61; del buf61  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf90, 42, grid=grid(42), stream=stream0)
        buf91 = buf60; del buf60  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf91, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf87, [None, None, reinterpret_tensor(buf90, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf91], buf89, True)
        buf94 = reinterpret_tensor(buf83, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf83  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf87, buf94, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf95 = reinterpret_tensor(buf77, (6272, 192), (192, 1), 0); del buf77  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (6272, 192), (192, 1), 0), reinterpret_tensor(arg44_1, (192, 192), (1, 192), 0), out=buf95)
        del arg44_1
        buf99 = buf94; del buf94  # reuse
        # Source Nodes: [getattr_l__mod___network_0___2___norm2, x_29, x_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf73, buf95, arg45_1, arg46_1, arg47_1, buf99, 6272, 192, grid=grid(6272), stream=stream0)
        del arg46_1
        del arg47_1
        buf100 = reinterpret_tensor(buf71, (6272, 576), (576, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (6272, 192), (192, 1), 0), reinterpret_tensor(arg48_1, (192, 576), (1, 192), 0), out=buf100)
        del arg48_1
        buf101 = reinterpret_tensor(buf100, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf100  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf101, arg49_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg49_1
        buf102 = reinterpret_tensor(buf99, (6272, 192), (192, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf101, (6272, 576), (576, 1), 0), reinterpret_tensor(arg50_1, (576, 192), (1, 576), 0), out=buf102)
        del arg50_1
        buf103 = reinterpret_tensor(buf102, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf102  # reuse
        buf107 = buf43; del buf43  # reuse
        # Source Nodes: [getattr_l__mod___network_0___3___norm1, x_29, x_31, x_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf103, buf73, buf95, arg45_1, arg51_1, arg52_1, arg53_1, buf107, 6272, 192, grid=grid(6272), stream=stream0)
        del arg45_1
        del arg51_1
        del arg52_1
        del arg53_1
        buf108 = reinterpret_tensor(buf79, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf79  # reuse
        # Source Nodes: [getattr_l__mod___network_0___3___attn_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_18.run(buf107, buf108, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf109 = reinterpret_tensor(buf78, (1568, 192), (1, 1568), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf108, buf109, 301056, grid=grid(301056), stream=stream0)
        del buf108
        buf110 = reinterpret_tensor(buf84, (1568, 486), (486, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf109, reinterpret_tensor(arg55_1, (192, 486), (1, 192), 0), out=buf110)
        del arg55_1
        del buf109
        buf114 = reinterpret_tensor(buf80, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf80  # reuse
        # Source Nodes: [attn_17, attn_18], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_6.run(buf110, arg56_1, buf114, 84672, 9, grid=grid(84672), stream=stream0)
        del arg56_1
        del buf110
        buf113 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_l__mod___network_0___3___attn_v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (6272, 192), (192, 1), 0), reinterpret_tensor(arg54_1, (192, 192), (1, 192), 0), out=buf113)
        del arg54_1
        buf115 = reinterpret_tensor(buf89, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf89  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf113, buf115, 2709504, grid=grid(2709504), stream=stream0)
        buf116 = buf86; del buf86  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf115, (9408, 9, 32), (288, 32, 1), 0), out=buf116)
        del buf114
        buf117 = buf87; del buf87  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_9.run(buf117, 1382400, grid=grid(1382400), stream=stream0)
        buf118 = reinterpret_tensor(buf115, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf115  # reuse
        buf119 = reinterpret_tensor(buf118, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf118  # reuse
        # Source Nodes: [x_39, x_40], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_10.run(buf119, buf116, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        del buf116
        buf120 = buf91; del buf91  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf120, 42, grid=grid(42), stream=stream0)
        buf121 = buf90; del buf90  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_11.run(buf121, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf117, [None, None, reinterpret_tensor(buf120, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf121], buf119, True)
        del buf119
        del buf120
        del buf121
        buf124 = reinterpret_tensor(buf113, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf113  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf117, buf124, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf117
        buf125 = reinterpret_tensor(buf107, (6272, 192), (192, 1), 0); del buf107  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (6272, 192), (192, 1), 0), reinterpret_tensor(arg57_1, (192, 192), (1, 192), 0), out=buf125)
        del arg57_1
        buf129 = buf124; del buf124  # reuse
        # Source Nodes: [getattr_l__mod___network_0___3___norm2, x_41, x_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf103, buf125, arg58_1, arg59_1, arg60_1, buf129, 6272, 192, grid=grid(6272), stream=stream0)
        del arg59_1
        del arg60_1
        buf130 = reinterpret_tensor(buf101, (6272, 576), (576, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (6272, 192), (192, 1), 0), reinterpret_tensor(arg61_1, (192, 576), (1, 192), 0), out=buf130)
        del arg61_1
        buf131 = reinterpret_tensor(buf130, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf130  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_16.run(buf131, arg62_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg62_1
        buf132 = reinterpret_tensor(buf129, (6272, 192), (192, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf131, (6272, 576), (576, 1), 0), reinterpret_tensor(arg63_1, (576, 192), (1, 576), 0), out=buf132)
        del arg63_1
        del buf131
        buf133 = reinterpret_tensor(buf73, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf73  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf103, buf125, arg58_1, buf132, arg64_1, buf133, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg58_1
        del arg64_1
        del buf103
        del buf125
        del buf132
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg65_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg65_1
        del buf133
        buf135 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf136 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_22.run(buf134, arg66_1, arg0_1, buf135, buf136, buf137, 4704, 128, grid=grid(4704), stream=stream0)
        buf138 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        buf139 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_23.run(buf135, buf136, buf137, buf138, buf139, 1568, 3, grid=grid(1568), stream=stream0)
        del buf135
        del buf136
        del buf137
        buf141 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___norm1, x_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_24.run(buf134, arg66_1, arg0_1, buf138, buf139, arg67_1, arg68_1, buf141, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg67_1
        del arg68_1
        del buf138
        del buf139
        buf142 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___network_2___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1568, 384), (384, 1), 0), reinterpret_tensor(arg69_1, (384, 1152), (1, 384), 0), out=buf142)
        del arg69_1
        buf143 = reinterpret_tensor(buf141, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf141  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf142, buf143, 602112, grid=grid(602112), stream=stream0)
        buf144 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf142, buf144, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf145 = empty((96, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf144, (96, 32, 196), (6272, 196, 1), 0), out=buf145)
        buf148 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20, attn_21], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf145, buf148, 18816, 196, grid=grid(18816), stream=stream0)
        buf149 = reinterpret_tensor(buf144, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf144  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf142, buf149, 602112, grid=grid(602112), stream=stream0)
        buf150 = reinterpret_tensor(buf143, (96, 196, 32), (6272, 32, 1), 0); del buf143  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf149, (96, 196, 32), (6272, 32, 1), 0), out=buf150)
        buf151 = reinterpret_tensor(buf149, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf149  # reuse
        # Source Nodes: [x_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf150, buf151, 602112, grid=grid(602112), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (1568, 384), (384, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (1568, 384), (384, 1), 0), reinterpret_tensor(arg70_1, (384, 384), (1, 384), 0), out=buf152)
        del arg70_1
        buf153 = reinterpret_tensor(buf152, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf152  # reuse
        buf157 = reinterpret_tensor(buf151, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf151  # reuse
        # Source Nodes: [getattr_l__mod___network_2___0___norm2, x_56, x_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_30.run(buf153, buf134, arg66_1, arg0_1, arg71_1, arg72_1, arg73_1, buf157, 1568, 384, grid=grid(1568), stream=stream0)
        del arg0_1
        del arg66_1
        del arg71_1
        del arg72_1
        del arg73_1
        buf158 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (1568, 384), (384, 1), 0), reinterpret_tensor(arg74_1, (384, 1152), (1, 384), 0), out=buf158)
        del arg74_1
        buf159 = reinterpret_tensor(buf158, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf158  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf159, arg75_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg75_1
        buf160 = reinterpret_tensor(buf157, (1568, 384), (384, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg76_1, (1152, 384), (1, 1152), 0), out=buf160)
        del arg76_1
        buf164 = reinterpret_tensor(buf134, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf134  # reuse
        # Source Nodes: [getattr_l__mod___network_2___1___norm1, x_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf153, buf160, arg77_1, arg78_1, arg79_1, buf164, 1568, 384, grid=grid(1568), stream=stream0)
        del arg78_1
        del arg79_1
        buf165 = reinterpret_tensor(buf159, (1568, 1152), (1152, 1), 0); del buf159  # reuse
        # Source Nodes: [getattr_l__mod___network_2___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 384), (384, 1), 0), reinterpret_tensor(arg80_1, (384, 1152), (1, 384), 0), out=buf165)
        del arg80_1
        buf166 = reinterpret_tensor(buf164, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf164  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf165, buf166, 602112, grid=grid(602112), stream=stream0)
        buf167 = empty((8, 12, 32, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf165, buf167, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf168 = reinterpret_tensor(buf148, (96, 196, 196), (38416, 196, 1), 0); del buf148  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf167, (96, 32, 196), (6272, 196, 1), 0), out=buf168)
        buf171 = reinterpret_tensor(buf145, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf145  # reuse
        # Source Nodes: [attn_23, attn_24], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf168, buf171, 18816, 196, grid=grid(18816), stream=stream0)
        buf172 = reinterpret_tensor(buf167, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf167  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf165, buf172, 602112, grid=grid(602112), stream=stream0)
        buf173 = reinterpret_tensor(buf166, (96, 196, 32), (6272, 32, 1), 0); del buf166  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf171, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf172, (96, 196, 32), (6272, 32, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf172, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf172  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf173, buf174, 602112, grid=grid(602112), stream=stream0)
        buf175 = reinterpret_tensor(buf173, (1568, 384), (384, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1568, 384), (384, 1), 0), reinterpret_tensor(arg81_1, (384, 384), (1, 384), 0), out=buf175)
        del arg81_1
        buf176 = reinterpret_tensor(buf175, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf175  # reuse
        buf180 = reinterpret_tensor(buf174, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf174  # reuse
        # Source Nodes: [getattr_l__mod___network_2___1___norm2, x_68, x_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf176, buf153, buf160, arg77_1, arg82_1, arg83_1, arg84_1, buf180, 1568, 384, grid=grid(1568), stream=stream0)
        del arg77_1
        del arg82_1
        del arg83_1
        del arg84_1
        buf181 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (1568, 384), (384, 1), 0), reinterpret_tensor(arg85_1, (384, 1152), (1, 384), 0), out=buf181)
        del arg85_1
        buf182 = reinterpret_tensor(buf181, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf181  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf182, arg86_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg86_1
        buf183 = reinterpret_tensor(buf180, (1568, 384), (384, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg87_1, (1152, 384), (1, 1152), 0), out=buf183)
        del arg87_1
        buf187 = reinterpret_tensor(buf160, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf160  # reuse
        # Source Nodes: [getattr_l__mod___network_2___2___norm1, x_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf176, buf183, arg88_1, arg89_1, arg90_1, buf187, 1568, 384, grid=grid(1568), stream=stream0)
        del arg89_1
        del arg90_1
        buf188 = reinterpret_tensor(buf182, (1568, 1152), (1152, 1), 0); del buf182  # reuse
        # Source Nodes: [getattr_l__mod___network_2___2___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 1152), (1, 384), 0), out=buf188)
        del arg91_1
        buf189 = reinterpret_tensor(buf187, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf187  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf188, buf189, 602112, grid=grid(602112), stream=stream0)
        buf190 = reinterpret_tensor(buf153, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf153  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf188, buf190, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf191 = reinterpret_tensor(buf171, (96, 196, 196), (38416, 196, 1), 0); del buf171  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf190, (96, 32, 196), (6272, 196, 1), 0), out=buf191)
        buf194 = reinterpret_tensor(buf168, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf168  # reuse
        # Source Nodes: [attn_26, attn_27], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf191, buf194, 18816, 196, grid=grid(18816), stream=stream0)
        buf195 = reinterpret_tensor(buf190, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf190  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf188, buf195, 602112, grid=grid(602112), stream=stream0)
        buf196 = reinterpret_tensor(buf189, (96, 196, 32), (6272, 32, 1), 0); del buf189  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf195, (96, 196, 32), (6272, 32, 1), 0), out=buf196)
        buf197 = reinterpret_tensor(buf195, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf195  # reuse
        # Source Nodes: [x_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf196, buf197, 602112, grid=grid(602112), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (1568, 384), (384, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (1568, 384), (384, 1), 0), reinterpret_tensor(arg92_1, (384, 384), (1, 384), 0), out=buf198)
        del arg92_1
        buf199 = reinterpret_tensor(buf198, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf198  # reuse
        buf203 = reinterpret_tensor(buf197, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf197  # reuse
        # Source Nodes: [getattr_l__mod___network_2___2___norm2, x_79, x_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf199, buf176, buf183, arg88_1, arg93_1, arg94_1, arg95_1, buf203, 1568, 384, grid=grid(1568), stream=stream0)
        del arg88_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf204 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (1568, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 1152), (1, 384), 0), out=buf204)
        del arg96_1
        buf205 = reinterpret_tensor(buf204, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf204  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf205, arg97_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg97_1
        buf206 = reinterpret_tensor(buf203, (1568, 384), (384, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg98_1, (1152, 384), (1, 1152), 0), out=buf206)
        del arg98_1
        buf210 = reinterpret_tensor(buf183, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf183  # reuse
        # Source Nodes: [getattr_l__mod___network_2___3___norm1, x_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf199, buf206, arg99_1, arg100_1, arg101_1, buf210, 1568, 384, grid=grid(1568), stream=stream0)
        del arg100_1
        del arg101_1
        buf211 = reinterpret_tensor(buf205, (1568, 1152), (1152, 1), 0); del buf205  # reuse
        # Source Nodes: [getattr_l__mod___network_2___3___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (1568, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 1152), (1, 384), 0), out=buf211)
        del arg102_1
        buf212 = reinterpret_tensor(buf210, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf210  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf211, buf212, 602112, grid=grid(602112), stream=stream0)
        buf213 = reinterpret_tensor(buf176, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf176  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf211, buf213, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf214 = reinterpret_tensor(buf194, (96, 196, 196), (38416, 196, 1), 0); del buf194  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf213, (96, 32, 196), (6272, 196, 1), 0), out=buf214)
        buf217 = reinterpret_tensor(buf191, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf191  # reuse
        # Source Nodes: [attn_29, attn_30], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf214, buf217, 18816, 196, grid=grid(18816), stream=stream0)
        buf218 = reinterpret_tensor(buf213, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf213  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf211, buf218, 602112, grid=grid(602112), stream=stream0)
        buf219 = reinterpret_tensor(buf212, (96, 196, 32), (6272, 32, 1), 0); del buf212  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf218, (96, 196, 32), (6272, 32, 1), 0), out=buf219)
        buf220 = reinterpret_tensor(buf218, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf218  # reuse
        # Source Nodes: [x_91], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf219, buf220, 602112, grid=grid(602112), stream=stream0)
        buf221 = reinterpret_tensor(buf219, (1568, 384), (384, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (1568, 384), (384, 1), 0), reinterpret_tensor(arg103_1, (384, 384), (1, 384), 0), out=buf221)
        del arg103_1
        buf222 = reinterpret_tensor(buf221, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf221  # reuse
        buf226 = reinterpret_tensor(buf220, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf220  # reuse
        # Source Nodes: [getattr_l__mod___network_2___3___norm2, x_90, x_94], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf222, buf199, buf206, arg99_1, arg104_1, arg105_1, arg106_1, buf226, 1568, 384, grid=grid(1568), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg99_1
        buf227 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1568, 384), (384, 1), 0), reinterpret_tensor(arg107_1, (384, 1152), (1, 384), 0), out=buf227)
        del arg107_1
        buf228 = reinterpret_tensor(buf227, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf227  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf228, arg108_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg108_1
        buf229 = reinterpret_tensor(buf226, (1568, 384), (384, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg109_1, (1152, 384), (1, 1152), 0), out=buf229)
        del arg109_1
        buf233 = reinterpret_tensor(buf206, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf206  # reuse
        # Source Nodes: [getattr_l__mod___network_3___0___norm1, x_102], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf222, buf229, arg110_1, arg111_1, arg112_1, buf233, 1568, 384, grid=grid(1568), stream=stream0)
        del arg111_1
        del arg112_1
        buf234 = reinterpret_tensor(buf228, (1568, 1152), (1152, 1), 0); del buf228  # reuse
        # Source Nodes: [getattr_l__mod___network_3___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1568, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 1152), (1, 384), 0), out=buf234)
        del arg113_1
        buf235 = reinterpret_tensor(buf233, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf233  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf234, buf235, 602112, grid=grid(602112), stream=stream0)
        buf236 = reinterpret_tensor(buf199, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf199  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf234, buf236, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf237 = reinterpret_tensor(buf217, (96, 196, 196), (38416, 196, 1), 0); del buf217  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf236, (96, 32, 196), (6272, 196, 1), 0), out=buf237)
        buf240 = reinterpret_tensor(buf214, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf214  # reuse
        # Source Nodes: [attn_32, attn_33], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf237, buf240, 18816, 196, grid=grid(18816), stream=stream0)
        buf241 = reinterpret_tensor(buf236, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf236  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf234, buf241, 602112, grid=grid(602112), stream=stream0)
        buf242 = reinterpret_tensor(buf235, (96, 196, 32), (6272, 32, 1), 0); del buf235  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf241, (96, 196, 32), (6272, 32, 1), 0), out=buf242)
        buf243 = reinterpret_tensor(buf241, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf241  # reuse
        # Source Nodes: [x_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf242, buf243, 602112, grid=grid(602112), stream=stream0)
        buf244 = reinterpret_tensor(buf242, (1568, 384), (384, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (1568, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 384), (1, 384), 0), out=buf244)
        del arg114_1
        buf245 = reinterpret_tensor(buf244, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf244  # reuse
        buf249 = reinterpret_tensor(buf243, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf243  # reuse
        # Source Nodes: [getattr_l__mod___network_3___0___norm2, x_102, x_106], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf245, buf222, buf229, arg110_1, arg115_1, arg116_1, arg117_1, buf249, 1568, 384, grid=grid(1568), stream=stream0)
        del arg110_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf250 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (1568, 384), (384, 1), 0), reinterpret_tensor(arg118_1, (384, 1152), (1, 384), 0), out=buf250)
        del arg118_1
        buf251 = reinterpret_tensor(buf250, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf250  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf251, arg119_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg119_1
        buf252 = reinterpret_tensor(buf249, (1568, 384), (384, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf251, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg120_1, (1152, 384), (1, 1152), 0), out=buf252)
        del arg120_1
        buf256 = reinterpret_tensor(buf229, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf229  # reuse
        # Source Nodes: [getattr_l__mod___network_3___1___norm1, x_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf245, buf252, arg121_1, arg122_1, arg123_1, buf256, 1568, 384, grid=grid(1568), stream=stream0)
        del arg122_1
        del arg123_1
        buf257 = reinterpret_tensor(buf251, (1568, 1152), (1152, 1), 0); del buf251  # reuse
        # Source Nodes: [getattr_l__mod___network_3___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (1568, 384), (384, 1), 0), reinterpret_tensor(arg124_1, (384, 1152), (1, 384), 0), out=buf257)
        del arg124_1
        buf258 = reinterpret_tensor(buf256, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf256  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf257, buf258, 602112, grid=grid(602112), stream=stream0)
        buf259 = reinterpret_tensor(buf222, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf222  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf257, buf259, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf260 = reinterpret_tensor(buf240, (96, 196, 196), (38416, 196, 1), 0); del buf240  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf259, (96, 32, 196), (6272, 196, 1), 0), out=buf260)
        buf263 = reinterpret_tensor(buf237, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf237  # reuse
        # Source Nodes: [attn_35, attn_36], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf260, buf263, 18816, 196, grid=grid(18816), stream=stream0)
        buf264 = reinterpret_tensor(buf259, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf259  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf257, buf264, 602112, grid=grid(602112), stream=stream0)
        buf265 = reinterpret_tensor(buf258, (96, 196, 32), (6272, 32, 1), 0); del buf258  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf264, (96, 196, 32), (6272, 32, 1), 0), out=buf265)
        buf266 = reinterpret_tensor(buf264, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf264  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf265, buf266, 602112, grid=grid(602112), stream=stream0)
        buf267 = reinterpret_tensor(buf265, (1568, 384), (384, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (1568, 384), (384, 1), 0), reinterpret_tensor(arg125_1, (384, 384), (1, 384), 0), out=buf267)
        del arg125_1
        buf268 = reinterpret_tensor(buf267, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf267  # reuse
        buf272 = reinterpret_tensor(buf266, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf266  # reuse
        # Source Nodes: [getattr_l__mod___network_3___1___norm2, x_113, x_117], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf268, buf245, buf252, arg121_1, arg126_1, arg127_1, arg128_1, buf272, 1568, 384, grid=grid(1568), stream=stream0)
        del arg121_1
        del arg126_1
        del arg127_1
        del arg128_1
        buf273 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (1568, 384), (384, 1), 0), reinterpret_tensor(arg129_1, (384, 1152), (1, 384), 0), out=buf273)
        del arg129_1
        buf274 = reinterpret_tensor(buf273, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf273  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf274, arg130_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg130_1
        buf275 = reinterpret_tensor(buf272, (1568, 384), (384, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg131_1, (1152, 384), (1, 1152), 0), out=buf275)
        del arg131_1
        buf279 = reinterpret_tensor(buf252, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf252  # reuse
        # Source Nodes: [getattr_l__mod___network_3___2___norm1, x_124], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf268, buf275, arg132_1, arg133_1, arg134_1, buf279, 1568, 384, grid=grid(1568), stream=stream0)
        del arg133_1
        del arg134_1
        buf280 = reinterpret_tensor(buf274, (1568, 1152), (1152, 1), 0); del buf274  # reuse
        # Source Nodes: [getattr_l__mod___network_3___2___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (1568, 384), (384, 1), 0), reinterpret_tensor(arg135_1, (384, 1152), (1, 384), 0), out=buf280)
        del arg135_1
        buf281 = reinterpret_tensor(buf279, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf279  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf280, buf281, 602112, grid=grid(602112), stream=stream0)
        buf282 = reinterpret_tensor(buf245, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf245  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf280, buf282, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf283 = reinterpret_tensor(buf263, (96, 196, 196), (38416, 196, 1), 0); del buf263  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf281, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf282, (96, 32, 196), (6272, 196, 1), 0), out=buf283)
        buf286 = reinterpret_tensor(buf260, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf260  # reuse
        # Source Nodes: [attn_38, attn_39], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf283, buf286, 18816, 196, grid=grid(18816), stream=stream0)
        buf287 = reinterpret_tensor(buf282, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf282  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf280, buf287, 602112, grid=grid(602112), stream=stream0)
        buf288 = reinterpret_tensor(buf281, (96, 196, 32), (6272, 32, 1), 0); del buf281  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf286, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf287, (96, 196, 32), (6272, 32, 1), 0), out=buf288)
        buf289 = reinterpret_tensor(buf287, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf287  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf288, buf289, 602112, grid=grid(602112), stream=stream0)
        buf290 = reinterpret_tensor(buf288, (1568, 384), (384, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf289, (1568, 384), (384, 1), 0), reinterpret_tensor(arg136_1, (384, 384), (1, 384), 0), out=buf290)
        del arg136_1
        buf291 = reinterpret_tensor(buf290, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf290  # reuse
        buf295 = reinterpret_tensor(buf289, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf289  # reuse
        # Source Nodes: [getattr_l__mod___network_3___2___norm2, x_124, x_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf291, buf268, buf275, arg132_1, arg137_1, arg138_1, arg139_1, buf295, 1568, 384, grid=grid(1568), stream=stream0)
        del arg132_1
        del arg137_1
        del arg138_1
        del arg139_1
        buf296 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf295, (1568, 384), (384, 1), 0), reinterpret_tensor(arg140_1, (384, 1152), (1, 384), 0), out=buf296)
        del arg140_1
        buf297 = reinterpret_tensor(buf296, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf296  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf297, arg141_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg141_1
        buf298 = reinterpret_tensor(buf295, (1568, 384), (384, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg142_1, (1152, 384), (1, 1152), 0), out=buf298)
        del arg142_1
        buf302 = reinterpret_tensor(buf275, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf275  # reuse
        # Source Nodes: [getattr_l__mod___network_3___3___norm1, x_135], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf291, buf298, arg143_1, arg144_1, arg145_1, buf302, 1568, 384, grid=grid(1568), stream=stream0)
        del arg144_1
        del arg145_1
        buf303 = reinterpret_tensor(buf297, (1568, 1152), (1152, 1), 0); del buf297  # reuse
        # Source Nodes: [getattr_l__mod___network_3___3___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (1568, 384), (384, 1), 0), reinterpret_tensor(arg146_1, (384, 1152), (1, 384), 0), out=buf303)
        del arg146_1
        buf304 = reinterpret_tensor(buf302, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf302  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf303, buf304, 602112, grid=grid(602112), stream=stream0)
        buf305 = reinterpret_tensor(buf268, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf268  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf303, buf305, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf306 = reinterpret_tensor(buf286, (96, 196, 196), (38416, 196, 1), 0); del buf286  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf305, (96, 32, 196), (6272, 196, 1), 0), out=buf306)
        buf309 = reinterpret_tensor(buf283, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf283  # reuse
        # Source Nodes: [attn_41, attn_42], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf306, buf309, 18816, 196, grid=grid(18816), stream=stream0)
        buf310 = reinterpret_tensor(buf305, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf305  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf303, buf310, 602112, grid=grid(602112), stream=stream0)
        buf311 = reinterpret_tensor(buf304, (96, 196, 32), (6272, 32, 1), 0); del buf304  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf310, (96, 196, 32), (6272, 32, 1), 0), out=buf311)
        buf312 = reinterpret_tensor(buf310, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf310  # reuse
        # Source Nodes: [x_136], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf311, buf312, 602112, grid=grid(602112), stream=stream0)
        buf313 = reinterpret_tensor(buf311, (1568, 384), (384, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (1568, 384), (384, 1), 0), reinterpret_tensor(arg147_1, (384, 384), (1, 384), 0), out=buf313)
        del arg147_1
        buf314 = reinterpret_tensor(buf313, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf313  # reuse
        buf318 = reinterpret_tensor(buf312, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf312  # reuse
        # Source Nodes: [getattr_l__mod___network_3___3___norm2, x_135, x_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf314, buf291, buf298, arg143_1, arg148_1, arg149_1, arg150_1, buf318, 1568, 384, grid=grid(1568), stream=stream0)
        del arg143_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf319 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf318, (1568, 384), (384, 1), 0), reinterpret_tensor(arg151_1, (384, 1152), (1, 384), 0), out=buf319)
        del arg151_1
        buf320 = reinterpret_tensor(buf319, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf319  # reuse
        # Source Nodes: [x_141], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf320, arg152_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg152_1
        buf321 = reinterpret_tensor(buf318, (1568, 384), (384, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf320, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg153_1, (1152, 384), (1, 1152), 0), out=buf321)
        del arg153_1
        buf325 = reinterpret_tensor(buf298, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf298  # reuse
        # Source Nodes: [getattr_l__mod___network_3___4___norm1, x_146], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf314, buf321, arg154_1, arg155_1, arg156_1, buf325, 1568, 384, grid=grid(1568), stream=stream0)
        del arg155_1
        del arg156_1
        buf326 = reinterpret_tensor(buf320, (1568, 1152), (1152, 1), 0); del buf320  # reuse
        # Source Nodes: [getattr_l__mod___network_3___4___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (1568, 384), (384, 1), 0), reinterpret_tensor(arg157_1, (384, 1152), (1, 384), 0), out=buf326)
        del arg157_1
        buf327 = reinterpret_tensor(buf325, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf325  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf326, buf327, 602112, grid=grid(602112), stream=stream0)
        buf328 = reinterpret_tensor(buf291, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf291  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf326, buf328, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf329 = reinterpret_tensor(buf309, (96, 196, 196), (38416, 196, 1), 0); del buf309  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf327, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf328, (96, 32, 196), (6272, 196, 1), 0), out=buf329)
        buf332 = reinterpret_tensor(buf306, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf306  # reuse
        # Source Nodes: [attn_44, attn_45], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf329, buf332, 18816, 196, grid=grid(18816), stream=stream0)
        buf333 = reinterpret_tensor(buf328, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf328  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf326, buf333, 602112, grid=grid(602112), stream=stream0)
        buf334 = reinterpret_tensor(buf327, (96, 196, 32), (6272, 32, 1), 0); del buf327  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf333, (96, 196, 32), (6272, 32, 1), 0), out=buf334)
        buf335 = reinterpret_tensor(buf333, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf333  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf334, buf335, 602112, grid=grid(602112), stream=stream0)
        buf336 = reinterpret_tensor(buf334, (1568, 384), (384, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 384), (384, 1), 0), reinterpret_tensor(arg158_1, (384, 384), (1, 384), 0), out=buf336)
        del arg158_1
        buf337 = reinterpret_tensor(buf336, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf336  # reuse
        buf341 = reinterpret_tensor(buf335, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf335  # reuse
        # Source Nodes: [getattr_l__mod___network_3___4___norm2, x_146, x_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf337, buf314, buf321, arg154_1, arg159_1, arg160_1, arg161_1, buf341, 1568, 384, grid=grid(1568), stream=stream0)
        del arg154_1
        del arg159_1
        del arg160_1
        del arg161_1
        buf342 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (1568, 384), (384, 1), 0), reinterpret_tensor(arg162_1, (384, 1152), (1, 384), 0), out=buf342)
        del arg162_1
        buf343 = reinterpret_tensor(buf342, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf342  # reuse
        # Source Nodes: [x_152], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf343, arg163_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg163_1
        buf344 = reinterpret_tensor(buf341, (1568, 384), (384, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg164_1, (1152, 384), (1, 1152), 0), out=buf344)
        del arg164_1
        buf348 = reinterpret_tensor(buf321, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf321  # reuse
        # Source Nodes: [getattr_l__mod___network_3___5___norm1, x_157], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf337, buf344, arg165_1, arg166_1, arg167_1, buf348, 1568, 384, grid=grid(1568), stream=stream0)
        del arg166_1
        del arg167_1
        buf349 = reinterpret_tensor(buf343, (1568, 1152), (1152, 1), 0); del buf343  # reuse
        # Source Nodes: [getattr_l__mod___network_3___5___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 384), (384, 1), 0), reinterpret_tensor(arg168_1, (384, 1152), (1, 384), 0), out=buf349)
        del arg168_1
        buf350 = reinterpret_tensor(buf348, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf348  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf349, buf350, 602112, grid=grid(602112), stream=stream0)
        buf351 = reinterpret_tensor(buf314, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf314  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf349, buf351, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf352 = reinterpret_tensor(buf332, (96, 196, 196), (38416, 196, 1), 0); del buf332  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf350, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf351, (96, 32, 196), (6272, 196, 1), 0), out=buf352)
        buf355 = reinterpret_tensor(buf329, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf329  # reuse
        # Source Nodes: [attn_47, attn_48], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf352, buf355, 18816, 196, grid=grid(18816), stream=stream0)
        buf356 = reinterpret_tensor(buf351, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf351  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf349, buf356, 602112, grid=grid(602112), stream=stream0)
        buf357 = reinterpret_tensor(buf350, (96, 196, 32), (6272, 32, 1), 0); del buf350  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf356, (96, 196, 32), (6272, 32, 1), 0), out=buf357)
        buf358 = reinterpret_tensor(buf356, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf356  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf357, buf358, 602112, grid=grid(602112), stream=stream0)
        buf359 = reinterpret_tensor(buf357, (1568, 384), (384, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (1568, 384), (384, 1), 0), reinterpret_tensor(arg169_1, (384, 384), (1, 384), 0), out=buf359)
        del arg169_1
        buf360 = reinterpret_tensor(buf359, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf359  # reuse
        buf364 = reinterpret_tensor(buf358, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf358  # reuse
        # Source Nodes: [getattr_l__mod___network_3___5___norm2, x_157, x_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf360, buf337, buf344, arg165_1, arg170_1, arg171_1, arg172_1, buf364, 1568, 384, grid=grid(1568), stream=stream0)
        del arg165_1
        del arg170_1
        del arg171_1
        del arg172_1
        buf365 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 384), (384, 1), 0), reinterpret_tensor(arg173_1, (384, 1152), (1, 384), 0), out=buf365)
        del arg173_1
        buf366 = reinterpret_tensor(buf365, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf365  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf366, arg174_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg174_1
        buf367 = reinterpret_tensor(buf364, (1568, 384), (384, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg175_1, (1152, 384), (1, 1152), 0), out=buf367)
        del arg175_1
        buf371 = reinterpret_tensor(buf344, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf344  # reuse
        # Source Nodes: [getattr_l__mod___network_3___6___norm1, x_168], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf360, buf367, arg176_1, arg177_1, arg178_1, buf371, 1568, 384, grid=grid(1568), stream=stream0)
        del arg177_1
        del arg178_1
        buf372 = reinterpret_tensor(buf366, (1568, 1152), (1152, 1), 0); del buf366  # reuse
        # Source Nodes: [getattr_l__mod___network_3___6___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (1568, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 1152), (1, 384), 0), out=buf372)
        del arg179_1
        buf373 = reinterpret_tensor(buf371, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf371  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf372, buf373, 602112, grid=grid(602112), stream=stream0)
        buf374 = reinterpret_tensor(buf337, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf337  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf372, buf374, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf375 = reinterpret_tensor(buf355, (96, 196, 196), (38416, 196, 1), 0); del buf355  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf373, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf374, (96, 32, 196), (6272, 196, 1), 0), out=buf375)
        buf378 = reinterpret_tensor(buf352, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf352  # reuse
        # Source Nodes: [attn_50, attn_51], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf375, buf378, 18816, 196, grid=grid(18816), stream=stream0)
        buf379 = reinterpret_tensor(buf374, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf374  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf372, buf379, 602112, grid=grid(602112), stream=stream0)
        buf380 = reinterpret_tensor(buf373, (96, 196, 32), (6272, 32, 1), 0); del buf373  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf379, (96, 196, 32), (6272, 32, 1), 0), out=buf380)
        buf381 = reinterpret_tensor(buf379, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf379  # reuse
        # Source Nodes: [x_169], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf380, buf381, 602112, grid=grid(602112), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (1568, 384), (384, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1568, 384), (384, 1), 0), reinterpret_tensor(arg180_1, (384, 384), (1, 384), 0), out=buf382)
        del arg180_1
        buf383 = reinterpret_tensor(buf382, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf382  # reuse
        buf387 = reinterpret_tensor(buf381, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf381  # reuse
        # Source Nodes: [getattr_l__mod___network_3___6___norm2, x_168, x_172], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf383, buf360, buf367, arg176_1, arg181_1, arg182_1, arg183_1, buf387, 1568, 384, grid=grid(1568), stream=stream0)
        del arg176_1
        del arg181_1
        del arg182_1
        del arg183_1
        buf388 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (1568, 384), (384, 1), 0), reinterpret_tensor(arg184_1, (384, 1152), (1, 384), 0), out=buf388)
        del arg184_1
        buf389 = reinterpret_tensor(buf388, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf388  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf389, arg185_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg185_1
        buf390 = reinterpret_tensor(buf387, (1568, 384), (384, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg186_1, (1152, 384), (1, 1152), 0), out=buf390)
        del arg186_1
        buf394 = reinterpret_tensor(buf367, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf367  # reuse
        # Source Nodes: [getattr_l__mod___network_3___7___norm1, x_179], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf383, buf390, arg187_1, arg188_1, arg189_1, buf394, 1568, 384, grid=grid(1568), stream=stream0)
        del arg188_1
        del arg189_1
        buf395 = reinterpret_tensor(buf389, (1568, 1152), (1152, 1), 0); del buf389  # reuse
        # Source Nodes: [getattr_l__mod___network_3___7___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 384), (384, 1), 0), reinterpret_tensor(arg190_1, (384, 1152), (1, 384), 0), out=buf395)
        del arg190_1
        buf396 = reinterpret_tensor(buf394, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf394  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf395, buf396, 602112, grid=grid(602112), stream=stream0)
        buf397 = reinterpret_tensor(buf360, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf360  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf395, buf397, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf398 = reinterpret_tensor(buf378, (96, 196, 196), (38416, 196, 1), 0); del buf378  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf397, (96, 32, 196), (6272, 196, 1), 0), out=buf398)
        buf401 = reinterpret_tensor(buf375, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf375  # reuse
        # Source Nodes: [attn_53, attn_54], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf398, buf401, 18816, 196, grid=grid(18816), stream=stream0)
        buf402 = reinterpret_tensor(buf397, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf397  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf395, buf402, 602112, grid=grid(602112), stream=stream0)
        buf403 = reinterpret_tensor(buf396, (96, 196, 32), (6272, 32, 1), 0); del buf396  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf402, (96, 196, 32), (6272, 32, 1), 0), out=buf403)
        buf404 = reinterpret_tensor(buf402, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf402  # reuse
        # Source Nodes: [x_180], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf403, buf404, 602112, grid=grid(602112), stream=stream0)
        buf405 = reinterpret_tensor(buf403, (1568, 384), (384, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (1568, 384), (384, 1), 0), reinterpret_tensor(arg191_1, (384, 384), (1, 384), 0), out=buf405)
        del arg191_1
        buf406 = reinterpret_tensor(buf405, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf405  # reuse
        buf410 = reinterpret_tensor(buf404, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf404  # reuse
        # Source Nodes: [getattr_l__mod___network_3___7___norm2, x_179, x_183], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf406, buf383, buf390, arg187_1, arg192_1, arg193_1, arg194_1, buf410, 1568, 384, grid=grid(1568), stream=stream0)
        del arg187_1
        del arg192_1
        del arg193_1
        del arg194_1
        buf411 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf410, (1568, 384), (384, 1), 0), reinterpret_tensor(arg195_1, (384, 1152), (1, 384), 0), out=buf411)
        del arg195_1
        buf412 = reinterpret_tensor(buf411, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf411  # reuse
        # Source Nodes: [x_185], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf412, arg196_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg196_1
        buf413 = reinterpret_tensor(buf410, (1568, 384), (384, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg197_1, (1152, 384), (1, 1152), 0), out=buf413)
        del arg197_1
        buf417 = reinterpret_tensor(buf390, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf390  # reuse
        # Source Nodes: [getattr_l__mod___network_4___0___norm1, x_191], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf406, buf413, arg198_1, arg199_1, arg200_1, buf417, 1568, 384, grid=grid(1568), stream=stream0)
        del arg199_1
        del arg200_1
        buf418 = reinterpret_tensor(buf412, (1568, 1152), (1152, 1), 0); del buf412  # reuse
        # Source Nodes: [getattr_l__mod___network_4___0___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 384), (384, 1), 0), reinterpret_tensor(arg201_1, (384, 1152), (1, 384), 0), out=buf418)
        del arg201_1
        buf419 = reinterpret_tensor(buf417, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf417  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf418, buf419, 602112, grid=grid(602112), stream=stream0)
        buf420 = reinterpret_tensor(buf383, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf383  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf418, buf420, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf421 = reinterpret_tensor(buf401, (96, 196, 196), (38416, 196, 1), 0); del buf401  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf420, (96, 32, 196), (6272, 196, 1), 0), out=buf421)
        buf424 = reinterpret_tensor(buf398, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf398  # reuse
        # Source Nodes: [attn_56, attn_57], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf421, buf424, 18816, 196, grid=grid(18816), stream=stream0)
        buf425 = reinterpret_tensor(buf420, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf420  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf418, buf425, 602112, grid=grid(602112), stream=stream0)
        buf426 = reinterpret_tensor(buf419, (96, 196, 32), (6272, 32, 1), 0); del buf419  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf425, (96, 196, 32), (6272, 32, 1), 0), out=buf426)
        buf427 = reinterpret_tensor(buf425, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf425  # reuse
        # Source Nodes: [x_192], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf426, buf427, 602112, grid=grid(602112), stream=stream0)
        buf428 = reinterpret_tensor(buf426, (1568, 384), (384, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf427, (1568, 384), (384, 1), 0), reinterpret_tensor(arg202_1, (384, 384), (1, 384), 0), out=buf428)
        del arg202_1
        buf429 = reinterpret_tensor(buf428, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf428  # reuse
        buf433 = reinterpret_tensor(buf427, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf427  # reuse
        # Source Nodes: [getattr_l__mod___network_4___0___norm2, x_191, x_195], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf429, buf406, buf413, arg198_1, arg203_1, arg204_1, arg205_1, buf433, 1568, 384, grid=grid(1568), stream=stream0)
        del arg198_1
        del arg203_1
        del arg204_1
        del arg205_1
        buf434 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf433, (1568, 384), (384, 1), 0), reinterpret_tensor(arg206_1, (384, 1152), (1, 384), 0), out=buf434)
        del arg206_1
        buf435 = reinterpret_tensor(buf434, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf434  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf435, arg207_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg207_1
        buf436 = reinterpret_tensor(buf433, (1568, 384), (384, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg208_1, (1152, 384), (1, 1152), 0), out=buf436)
        del arg208_1
        buf440 = reinterpret_tensor(buf413, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf413  # reuse
        # Source Nodes: [getattr_l__mod___network_4___1___norm1, x_202], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf429, buf436, arg209_1, arg210_1, arg211_1, buf440, 1568, 384, grid=grid(1568), stream=stream0)
        del arg210_1
        del arg211_1
        buf441 = reinterpret_tensor(buf435, (1568, 1152), (1152, 1), 0); del buf435  # reuse
        # Source Nodes: [getattr_l__mod___network_4___1___attn_qkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (1568, 384), (384, 1), 0), reinterpret_tensor(arg212_1, (384, 1152), (1, 384), 0), out=buf441)
        del arg212_1
        buf442 = reinterpret_tensor(buf440, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf440  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf441, buf442, 602112, grid=grid(602112), stream=stream0)
        buf443 = reinterpret_tensor(buf406, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf406  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_26.run(buf441, buf443, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf444 = reinterpret_tensor(buf424, (96, 196, 196), (38416, 196, 1), 0); del buf424  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf442, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf443, (96, 32, 196), (6272, 196, 1), 0), out=buf444)
        buf447 = reinterpret_tensor(buf421, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf421  # reuse
        # Source Nodes: [attn_59, attn_60], Original ATen: [aten._softmax, aten.mul]
        triton_per_fused__softmax_mul_27.run(buf444, buf447, 18816, 196, grid=grid(18816), stream=stream0)
        del buf444
        buf448 = reinterpret_tensor(buf443, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf443  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf441, buf448, 602112, grid=grid(602112), stream=stream0)
        buf449 = reinterpret_tensor(buf442, (96, 196, 32), (6272, 32, 1), 0); del buf442  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf448, (96, 196, 32), (6272, 32, 1), 0), out=buf449)
        del buf447
        buf450 = reinterpret_tensor(buf448, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf448  # reuse
        # Source Nodes: [x_203], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf449, buf450, 602112, grid=grid(602112), stream=stream0)
        buf451 = reinterpret_tensor(buf449, (1568, 384), (384, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 384), (384, 1), 0), reinterpret_tensor(arg213_1, (384, 384), (1, 384), 0), out=buf451)
        del arg213_1
        buf452 = reinterpret_tensor(buf451, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf451  # reuse
        buf456 = reinterpret_tensor(buf450, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf450  # reuse
        # Source Nodes: [getattr_l__mod___network_4___1___norm2, x_202, x_206], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf452, buf429, buf436, arg209_1, arg214_1, arg215_1, arg216_1, buf456, 1568, 384, grid=grid(1568), stream=stream0)
        del arg209_1
        del arg214_1
        del arg215_1
        del arg216_1
        del buf429
        del buf436
        buf457 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 384), (384, 1), 0), reinterpret_tensor(arg217_1, (384, 1152), (1, 384), 0), out=buf457)
        del arg217_1
        buf458 = reinterpret_tensor(buf457, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf457  # reuse
        # Source Nodes: [x_208], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_31.run(buf458, arg218_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg218_1
        buf459 = reinterpret_tensor(buf456, (1568, 384), (384, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg219_1, (1152, 384), (1, 1152), 0), out=buf459)
        del arg219_1
        del buf458
        buf460 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf461 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, l__mod___post_network_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_34.run(arg1_1, buf452, buf459, arg220_1, buf460, buf461, 1576, 384, grid=grid(1576), stream=stream0)
        buf463 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5, l__mod___post_network_0_norm1], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_35.run(arg1_1, buf452, buf459, arg220_1, buf460, buf461, arg221_1, arg222_1, buf463, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del arg221_1
        del arg222_1
        buf464 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_attn_kv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (1576, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 768), (1, 384), 0), out=buf464)
        del arg223_1
        buf465 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_attn_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (8, 384), (75648, 1), 0), reinterpret_tensor(arg224_1, (384, 384), (1, 384), 0), out=buf465)
        del arg224_1
        buf466 = reinterpret_tensor(buf465, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf465  # reuse
        # Source Nodes: [mul_18], Original ATen: [aten.mul]
        triton_poi_fused_mul_36.run(buf466, 3072, grid=grid(3072), stream=stream0)
        buf467 = reinterpret_tensor(buf463, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf463  # reuse
        # Source Nodes: [attn_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf464, buf467, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf468 = empty((96, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf467, (96, 32, 197), (6304, 197, 1), 0), out=buf468)
        buf471 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_38.run(buf468, buf471, 96, 197, grid=grid(96), stream=stream0)
        buf472 = reinterpret_tensor(buf467, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf467  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf464, buf472, 605184, grid=grid(605184), stream=stream0)
        buf473 = reinterpret_tensor(buf466, (96, 1, 32), (32, 32, 1), 0); del buf466  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf471, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf472, (96, 197, 32), (6304, 32, 1), 0), out=buf473)
        buf474 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf473, (8, 384), (384, 1), 0), reinterpret_tensor(arg225_1, (384, 384), (1, 384), 0), out=buf474)
        del arg225_1
        buf475 = reinterpret_tensor(buf474, (8, 1, 384), (384, 3072, 1), 0); del buf474  # reuse
        buf479 = reinterpret_tensor(buf473, (8, 1, 384), (384, 384, 1), 0); del buf473  # reuse
        # Source Nodes: [cls_embed_4, l__mod___post_network_0_norm2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_40.run(buf475, arg1_1, buf452, buf459, arg220_1, arg226_1, arg227_1, arg228_1, buf479, 8, 384, grid=grid(8), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        buf480 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf479, (8, 384), (384, 1), 0), reinterpret_tensor(arg229_1, (384, 1152), (1, 384), 0), out=buf480)
        del arg229_1
        buf481 = reinterpret_tensor(buf480, (8, 1, 1152), (1152, 1152, 1), 0); del buf480  # reuse
        # Source Nodes: [x_219], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_41.run(buf481, arg230_1, 9216, grid=grid(9216), stream=stream0)
        del arg230_1
        buf482 = reinterpret_tensor(buf479, (8, 384), (384, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf481, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg231_1, (1152, 384), (1, 1152), 0), out=buf482)
        del arg231_1
        buf483 = reinterpret_tensor(buf472, (8, 197, 384), (75648, 1, 197), 0); del buf472  # reuse
        # Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf475, buf482, arg232_1, arg1_1, buf452, buf459, arg220_1, buf483, 3072, 197, grid=grid(3072, 197), stream=stream0)
        del arg1_1
        del arg220_1
        del arg232_1
        del buf452
        buf484 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        buf485 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        buf486 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_43.run(buf483, buf484, buf485, buf486, 4728, 128, grid=grid(4728), stream=stream0)
        buf487 = buf461; del buf461  # reuse
        buf488 = buf460; del buf460  # reuse
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_44.run(buf484, buf485, buf486, buf487, buf488, 1576, 3, grid=grid(1576), stream=stream0)
        buf490 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_45.run(buf483, buf487, buf488, arg233_1, arg234_1, buf490, 1576, 384, grid=grid(1576, 384), stream=stream0)
        del arg233_1
        del arg234_1
        buf491 = buf464; del buf464  # reuse
        # Source Nodes: [l__mod___post_network_1_attn_kv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (1576, 384), (384, 1), 0), reinterpret_tensor(arg235_1, (384, 768), (1, 384), 0), out=buf491)
        del arg235_1
        buf492 = buf482; del buf482  # reuse
        # Source Nodes: [l__mod___post_network_1_attn_q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (8, 384), (75648, 1), 0), reinterpret_tensor(arg236_1, (384, 384), (1, 384), 0), out=buf492)
        del arg236_1
        buf493 = reinterpret_tensor(buf492, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf492  # reuse
        # Source Nodes: [mul_19], Original ATen: [aten.mul]
        triton_poi_fused_mul_36.run(buf493, 3072, grid=grid(3072), stream=stream0)
        buf494 = reinterpret_tensor(buf490, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf490  # reuse
        # Source Nodes: [attn_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_37.run(buf491, buf494, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf495 = reinterpret_tensor(buf471, (96, 1, 197), (197, 197, 1), 0); del buf471  # reuse
        # Source Nodes: [attn_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf493, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf494, (96, 32, 197), (6304, 197, 1), 0), out=buf495)
        buf498 = reinterpret_tensor(buf468, (8, 12, 1, 197), (2364, 197, 197, 1), 0); del buf468  # reuse
        # Source Nodes: [attn_66], Original ATen: [aten._softmax]
        triton_per_fused__softmax_38.run(buf495, buf498, 96, 197, grid=grid(96), stream=stream0)
        del buf495
        buf499 = reinterpret_tensor(buf494, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf494  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf491, buf499, 605184, grid=grid(605184), stream=stream0)
        del buf491
        buf500 = reinterpret_tensor(buf493, (96, 1, 32), (32, 32, 1), 0); del buf493  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf498, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf499, (96, 197, 32), (6304, 32, 1), 0), out=buf500)
        del buf498
        buf501 = reinterpret_tensor(buf475, (8, 384), (384, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (8, 384), (384, 1), 0), reinterpret_tensor(arg237_1, (384, 384), (1, 384), 0), out=buf501)
        del arg237_1
        buf502 = empty_strided((8, 1, 1, 3), (3, 24, 24, 1), device='cuda', dtype=torch.float32)
        buf503 = empty_strided((8, 1, 1, 3), (3, 24, 24, 1), device='cuda', dtype=torch.float32)
        buf504 = empty_strided((8, 1, 1, 3), (3, 24, 24, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_46.run(buf483, buf501, arg238_1, buf502, buf503, buf504, 24, 128, grid=grid(24), stream=stream0)
        buf505 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        buf506 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_47.run(buf502, buf503, buf504, buf505, buf506, 8, 3, grid=grid(8), stream=stream0)
        del buf502
        del buf503
        del buf504
        buf508 = reinterpret_tensor(buf500, (8, 1, 384), (384, 384, 1), 0); del buf500  # reuse
        # Source Nodes: [cls_embed_10, l__mod___post_network_1_norm2], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_48.run(buf483, buf501, arg238_1, buf505, buf506, arg239_1, arg240_1, buf508, 3072, grid=grid(3072), stream=stream0)
        del arg239_1
        del arg240_1
        del buf505
        del buf506
        buf509 = reinterpret_tensor(buf481, (8, 1152), (1152, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf508, (8, 384), (384, 1), 0), reinterpret_tensor(arg241_1, (384, 1152), (1, 384), 0), out=buf509)
        del arg241_1
        buf510 = reinterpret_tensor(buf509, (8, 1, 1152), (1152, 1152, 1), 0); del buf509  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_41.run(buf510, arg242_1, 9216, grid=grid(9216), stream=stream0)
        del arg242_1
        buf511 = reinterpret_tensor(buf508, (8, 384), (384, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf510, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg243_1, (1152, 384), (1, 1152), 0), out=buf511)
        del arg243_1
        del buf510
        buf512 = reinterpret_tensor(buf499, (8, 197, 384), (75648, 1, 197), 0); del buf499  # reuse
        # Source Nodes: [cat_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf483, buf501, arg238_1, buf511, arg244_1, buf512, 1576, 384, grid=grid(1576, 384), stream=stream0)
        del arg238_1
        del arg244_1
        del buf483
        del buf501
        buf513 = buf486; del buf486  # reuse
        buf514 = buf485; del buf485  # reuse
        buf515 = buf484; del buf484  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_50.run(buf512, buf513, buf514, buf515, 4728, 128, grid=grid(4728), stream=stream0)
        buf516 = buf488; del buf488  # reuse
        buf517 = buf487; del buf487  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_44.run(buf513, buf514, buf515, buf516, buf517, 1576, 3, grid=grid(1576), stream=stream0)
        del buf513
        del buf514
        del buf515
        buf519 = buf512; del buf512  # reuse
        # Source Nodes: [x_234], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_51.run(buf519, buf516, buf517, arg245_1, arg246_1, 605184, grid=grid(605184), stream=stream0)
        del arg245_1
        del arg246_1
        del buf516
        del buf517
        buf520 = reinterpret_tensor(buf459, (8, 196, 384), (75264, 384, 1), 0); del buf459  # reuse
        # Source Nodes: [aux], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf519, buf520, 1568, 384, grid=grid(1568, 384), stream=stream0)
        buf521 = empty((1568, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [aux], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (1568, 384), (384, 1), 0), reinterpret_tensor(arg249_1, (384, 1000), (1, 384), 0), out=buf521)
        del arg249_1
        del buf520
        buf522 = empty_strided((8, 1000, 2), (2000, 1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [aux, max_1], Original ATen: [aten.add, aten.max]
        triton_red_fused_add_max_53.run(buf521, arg250_1, buf522, 16000, 98, grid=grid(16000), stream=stream0)
        del arg250_1
        del buf521
        buf525 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_54.run(buf519, buf525, 3072, grid=grid(3072), stream=stream0)
        del buf519
        buf526 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf525, reinterpret_tensor(arg247_1, (384, 1000), (1, 384), 0), out=buf526)
        del arg247_1
        del buf525
        buf523 = empty((8, 1000), device='cuda', dtype=torch.float32)
        buf527 = buf523; del buf523  # reuse
        # Source Nodes: [aux, max_1, mul_20, x_236], Original ATen: [aten.add, aten.max, aten.mul]
        triton_per_fused_add_max_mul_55.run(buf527, buf522, buf526, arg248_1, 8000, 2, grid=grid(8000), stream=stream0)
        del arg248_1
        return (buf527, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg254_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg257_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg260_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
