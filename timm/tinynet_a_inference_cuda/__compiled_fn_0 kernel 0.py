
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


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5p44ud5whnms3szu7alzmaxxy6jjxb5p2jqjrx3qdcv5ts2gbm.py
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
    xnumel = 36864
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36864*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (110592*y1)), tmp0, ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbxnrzrqop4klqko3lxrugnst7kip4kb4m4tqgyf3ww2kcgb2l.py
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
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbvqlfhbrtyxtkud3sj5qg2yzs3dju7zdpgnrg4swtygxgqw2uzc.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# shortcut => mul_3, sigmoid
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (32*x2) + (294912*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55vsc3ow73hfpbolv4nuknxvqs4ikzs5im2xeqivwkogpe6fg5m.py
# Source Nodes: [x_6, x_9, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_6 => add_3, mul_5, mul_6, sub_1
# x_9 => mul_7, sigmoid_1
# x_se => mean
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 16384],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 32
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (9216*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp15 = tl.sigmoid(tmp14)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r2 + (9216*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 9216.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nspemircglvj5s24b47x4mudxqjfk7c2thagb5v6qb5c47gxe3.py
# Source Nodes: [x_9, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_9 => mul_7, sigmoid_1
# x_se => mean
# x_se_1 => convolution_2
# x_se_2 => mul_8, sigmoid_2
triton_poi_fused_convolution_mean_silu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_4', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3iwrlyrib24wb3bqnuc2qkcvojuzi6arndg33hlsrmgmwrck6e.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_9
# x_9 => mul_7, sigmoid_1
# x_se => mean
# x_se_1 => convolution_2
# x_se_2 => mul_8, sigmoid_2
# x_se_3 => convolution_3
triton_poi_fused_convolution_mean_mul_sigmoid_silu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (32*x2) + (294912*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbkq5ejkcsemxi7lyywerv2vh2ti6iig6o6dw2ct7skaervo533.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_12 => add_5, mul_11, mul_12, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 9216
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (147456*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7scdvjfbyz4j3fj7232pxejtubsjwvqshxwqjjaxmr7fwh5dnz.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_17 => add_7, mul_14, mul_15, sub_3
# x_20 => mul_16, sigmoid_4
triton_poi_fused__native_batch_norm_legit_no_training_silu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 9216
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (9216*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (96*x2) + (884736*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqen7akn6gpnqazfctvhld5ovmvdm2y5v25zzc7dsef6ly4ioat.py
# Source Nodes: [x_22, x_25, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_22 => add_9, mul_18, mul_19, sub_4
# x_25 => mul_20, sigmoid_5
# x_se_4 => mean_1
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 96
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (2304*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp15 = tl.sigmoid(tmp14)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r2 + (2304*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 2304.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbboxzc4fe2d2x7htio2dy5hj3vlrw6ika3knyyp5jycksssis2.py
# Source Nodes: [x_25, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_25 => mul_20, sigmoid_5
# x_se_4 => mean_1
# x_se_5 => convolution_7
# x_se_6 => mul_21, sigmoid_6
triton_poi_fused_convolution_mean_silu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwdd7xdaqeobotokfhwp4if6lboigpvxpunzkru3yyo4yk6ia7i.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_25 => mul_20, sigmoid_5
# x_26 => mul_22
# x_se_4 => mean_1
# x_se_5 => convolution_7
# x_se_6 => mul_21, sigmoid_6
# x_se_7 => convolution_8
triton_poi_fused_convolution_mean_mul_sigmoid_silu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 2304
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
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (96*x2) + (221184*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjsergxzs6hvccfaspbaaxirp5ndwfuhs4wowab46xrshh3ogpo3.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_28 => add_11, mul_24, mul_25, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (55296*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/caktoor7psvlxza2pxwj2yxzzii52km6rfpduluomo3su6wecqob.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_33 => add_13, mul_27, mul_28, sub_6
# x_36 => mul_29, sigmoid_8
triton_poi_fused__native_batch_norm_legit_no_training_silu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (144*x2) + (331776*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csaawagl5lxxnuytkq4u7rbmhpiik3teysese4irzt5nuxzcpm7i.py
# Source Nodes: [x_38, x_41, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_38 => add_15, mul_31, mul_32, sub_7
# x_41 => mul_33, sigmoid_9
# x_se_8 => mean_2
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 144
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (2304*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp15 = tl.sigmoid(tmp14)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r2 + (2304*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 2304.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjtpxqe4htgf4ddxxpale2frbwlkmsl77akx5qxu4cycgpnlevx.py
# Source Nodes: [x_41, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_41 => mul_33, sigmoid_9
# x_se_10 => mul_34, sigmoid_10
# x_se_8 => mean_2
# x_se_9 => convolution_12
triton_poi_fused_convolution_mean_silu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_14', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsyq7yzkqmr4wj7v2fzyjl6rrqpc5jpf35kpevtazckubcgw37o.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_41 => mul_33, sigmoid_9
# x_42 => mul_35
# x_se_10 => mul_34, sigmoid_10
# x_se_11 => convolution_13
# x_se_8 => mean_2
# x_se_9 => convolution_12
triton_poi_fused_convolution_mean_mul_sigmoid_silu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 2304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (2304*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (144*x2) + (331776*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v4/cv44c5ha5eqlyf4wssxvt6junxedv33gq7uqeyms3utsudt4al3w.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_44 => add_17, mul_37, mul_38, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2304
    y1 = (yindex // 2304)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2304*x2) + (55296*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/au/cauenwu22vqfd3cz3ikufdq4sxr62eajl23pjbgzrvhiic22n4fc.py
# Source Nodes: [x_55, x_58, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_55 => add_22, mul_44, mul_45, sub_10
# x_58 => mul_46, sigmoid_13
# x_se_12 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1152
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_out_ptr0 + (r2 + (576*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 576.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (576*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmk5dn2q3aqtwftta624txowcv4dspfsvvopji55pu5apdfh4hrg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_58 => mul_46, sigmoid_13
# x_59 => mul_48
# x_se_12 => mean_3
# x_se_13 => convolution_17
# x_se_14 => mul_47, sigmoid_14
# x_se_15 => convolution_18
triton_poi_fused_convolution_mean_mul_sigmoid_silu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (144*x2) + (82944*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsihqz7rffit5q66iswsl5slpmp6tvgxn2j76u35ylq44wzcpxa.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_61 => add_24, mul_50, mul_51, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (40*x2) + (23040*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstijpmj2twk4gvqrkzvkcncncjimxjsh6axdusoo3qso6vwqy5v.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_66 => add_26, mul_53, mul_54, sub_12
# x_69 => mul_55, sigmoid_16
triton_poi_fused__native_batch_norm_legit_no_training_silu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (240*x2) + (138240*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfjcgoqcueb47ojs7jht4chkx7ujrrkr375efggj3ewmriixjdg.py
# Source Nodes: [x_71, x_74, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_71 => add_28, mul_57, mul_58, sub_13
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1920
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (r2 + (576*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = 576.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (576*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pzgk4aknimktirijk4g2hzdfsyv5megb3ssn7b62bnt754a6mj.py
# Source Nodes: [x_74, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
triton_poi_fused_convolution_mean_silu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 10
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxpjf6h5w6gv5bvhp4ealx3um37xwub6d7fxj2x3rp4se3fr6w6n.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_74 => mul_59, sigmoid_17
# x_75 => mul_61
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
# x_se_19 => convolution_23
triton_poi_fused_convolution_mean_mul_sigmoid_silu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (576*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (240*x2) + (138240*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidekv53apwwge42pp5ptwspx2iojmmvg2fr4seeldl67uuss26n.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_31
# x_77 => add_30, mul_63, mul_64, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 40
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 576
    y1 = (yindex // 576)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (23040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (40*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (40*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbiyof336cinnzyiskcfana3nnw7jebar6ummwn7wtgyfsrr2y6d.py
# Source Nodes: [x_88, x_91, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_88 => add_35, mul_70, mul_71, sub_16
# x_91 => mul_72, sigmoid_21
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 144.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjvs2k2sb4i5rulnivcv737u6skghkbsxa5wt3oqmaxjwufi4s7e.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_91 => mul_72, sigmoid_21
# x_92 => mul_74
# x_se_20 => mean_5
# x_se_21 => convolution_27
# x_se_22 => mul_73, sigmoid_22
# x_se_23 => convolution_28
triton_poi_fused_convolution_mean_mul_sigmoid_silu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (240*x2) + (34560*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdiwqkz6aq4zo2qtncaq3wpwacfou6ho2owbmvyf6r4xyeysjvpd.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_94 => add_37, mul_76, mul_77, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (11520*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxv56i7yhkweo37cuxt42go37eblhnkp6vmnontlq5kg563sdmx.py
# Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_102 => mul_81, sigmoid_24
# x_99 => add_39, mul_79, mul_80, sub_18
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (480*x2) + (69120*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqjvt4nqc2krwl4gprl6sstkeji67vhjx6utoj54s2w3uvlsirj.py
# Source Nodes: [x_104, x_107, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_104 => add_41, mul_83, mul_84, sub_19
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 144.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gx/cgx5iw2r2mwthc5a6ctzpy3uvekuvi27pwfzzgrhg2fjc37vo5j7.py
# Source Nodes: [x_107, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
triton_poi_fused_convolution_mean_silu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 20
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgez7sp6cs4omxptjfqbulvrwlyctxva2c3pnzpc3o52w6efuumk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_107 => mul_85, sigmoid_25
# x_108 => mul_87
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
# x_se_27 => convolution_33
triton_poi_fused_convolution_mean_mul_sigmoid_silu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (480*x2) + (69120*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cuamtufivwpvha3vykuhxcf3frwjwcj35m7wxxof5bcqacgue7t3.py
# Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_7 => add_44
# x_110 => add_43, mul_89, mul_90, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_add_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 80
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (11520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (80*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clopx6mtdb7xe5uzts74yfd6qaqtfnaj3s6yic7ugh6cytxy6j6r.py
# Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_161 => add_64, mul_128, mul_129, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (112*x2) + (16128*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagw2uovwmhumcjlhisgnyrqvxqzfmdqjnlhduuqwq4or6yu56fx.py
# Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_166 => add_66, mul_131, mul_132, sub_30
# x_169 => mul_133, sigmoid_40
triton_poi_fused__native_batch_norm_legit_no_training_silu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (672*x2) + (96768*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vsljdn3ag3dgvqueadoyr7w33x6bwwmolapc3lynrr5rart3jk.py
# Source Nodes: [x_171, x_174, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_171 => add_68, mul_135, mul_136, sub_31
# x_174 => mul_137, sigmoid_41
# x_se_40 => mean_10
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 144.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjknn3lagmfo43yzalw3qy4acbjymqui4jqnyk32fvzdiclglza5.py
# Source Nodes: [x_174, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_174 => mul_137, sigmoid_41
# x_se_40 => mean_10
# x_se_41 => convolution_52
# x_se_42 => mul_138, sigmoid_42
triton_poi_fused_convolution_mean_silu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/clef2fwple6wyau73uh3sm5uwk4fwf2v47hwzmrbgud6cwtrp6wp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_43
# x_174 => mul_137, sigmoid_41
# x_175 => mul_139
# x_se_40 => mean_10
# x_se_41 => convolution_52
# x_se_42 => mul_138, sigmoid_42
# x_se_43 => convolution_53
triton_poi_fused_convolution_mean_mul_sigmoid_silu_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (144*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (672*x2) + (96768*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvda2o37ozf3wrfjdpjxbtv33rpx5rdawaxd2jgtp3q7n6muqys.py
# Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_11 => add_71
# x_177 => add_70, mul_141, mul_142, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_add_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 112
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (144*x2) + (16128*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (112*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5xaap7aofc7slkpsslh2baibh4klwel7q66xyexv66i6izag6k.py
# Source Nodes: [x_222, x_225, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_222 => add_89, mul_174, mul_175, sub_40
# x_225 => mul_176, sigmoid_53
# x_se_52 => mean_13
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (36*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 36.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (36*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotyt4opmsw4d2ahzkifn64klv2uxqaipvj7ylldpxhyto6xuaz6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_55
# x_225 => mul_176, sigmoid_53
# x_226 => mul_178
# x_se_52 => mean_13
# x_se_53 => convolution_67
# x_se_54 => mul_177, sigmoid_54
# x_se_55 => convolution_68
triton_poi_fused_convolution_mean_mul_sigmoid_silu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (672*x2) + (24192*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7k/c7kjxtkesmrdbb2tazfwwj4qbvtcqa7kovkk2hpbwneziyd5eg5s.py
# Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_228 => add_91, mul_180, mul_181, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 36
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
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (6912*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24ng4zryxuq43mhbq5egf7rcz2ywkbzcd5li433clo764r34xph.py
# Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_233 => add_93, mul_183, mul_184, sub_42
# x_236 => mul_185, sigmoid_56
triton_poi_fused__native_batch_norm_legit_no_training_silu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (36*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (y0 + (1152*x2) + (41472*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn57z3a7rukhoz2bfapoxfoqkow67ssszu4rzj65wchh3gg7rugu.py
# Source Nodes: [x_238, x_241, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_238 => add_95, mul_187, mul_188, sub_43
# x_241 => mul_189, sigmoid_57
# x_se_56 => mean_14
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (r2 + (36*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 36.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (36*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq37cowjf2thgbmcubpijglexfssc7je23wsxgajf5ug6t6pvjua.py
# Source Nodes: [x_241, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_241 => mul_189, sigmoid_57
# x_se_56 => mean_14
# x_se_57 => convolution_72
# x_se_58 => mul_190, sigmoid_58
triton_poi_fused_convolution_mean_silu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 48
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wzjr7n6sr2futyh7keefzm7rjjrrjochcnzakdwvljjtkeazk4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_59
# x_241 => mul_189, sigmoid_57
# x_242 => mul_191
# x_se_56 => mean_14
# x_se_57 => convolution_72
# x_se_58 => mul_190, sigmoid_58
# x_se_59 => convolution_73
triton_poi_fused_convolution_mean_mul_sigmoid_silu_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (1152*x2) + (41472*y1)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3o37pas5jvcf76ukg7to7iufeataa7ukcx7guacbn2ay4x665u.py
# Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_15 => add_98
# x_244 => add_97, mul_193, mul_194, sub_44
triton_poi_fused__native_batch_norm_legit_no_training_add_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (36*x2) + (6912*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfaym6qspsid4jwqzj6e3yhhpmfsxzllp7ilgok3ijdmhlp5ignf.py
# Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_312 => add_125, mul_245, mul_246, sub_56
triton_poi_fused__native_batch_norm_legit_no_training_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (36*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (320*x2) + (11520*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmn6qrl6of7bii27ushvu6ymi5gmqnmn2ft3aohbnaq2jw5p6me5.py
# Source Nodes: [x_318, x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_318 => add_127, mul_248, mul_249, sub_57
# x_322 => mul_250, sigmoid_76
# x_323 => mean_19
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_48', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (r2 + (36*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 36.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (24, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (144, ), (1, ))
    assert_size_stride(arg13_1, (144, ), (1, ))
    assert_size_stride(arg14_1, (144, ), (1, ))
    assert_size_stride(arg15_1, (144, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (144, ), (1, ))
    assert_size_stride(arg19_1, (144, ), (1, ))
    assert_size_stride(arg20_1, (144, ), (1, ))
    assert_size_stride(arg21_1, (144, ), (1, ))
    assert_size_stride(arg22_1, (40, ), (1, ))
    assert_size_stride(arg23_1, (40, ), (1, ))
    assert_size_stride(arg24_1, (240, ), (1, ))
    assert_size_stride(arg25_1, (240, ), (1, ))
    assert_size_stride(arg26_1, (240, ), (1, ))
    assert_size_stride(arg27_1, (240, ), (1, ))
    assert_size_stride(arg28_1, (40, ), (1, ))
    assert_size_stride(arg29_1, (40, ), (1, ))
    assert_size_stride(arg30_1, (240, ), (1, ))
    assert_size_stride(arg31_1, (240, ), (1, ))
    assert_size_stride(arg32_1, (240, ), (1, ))
    assert_size_stride(arg33_1, (240, ), (1, ))
    assert_size_stride(arg34_1, (80, ), (1, ))
    assert_size_stride(arg35_1, (80, ), (1, ))
    assert_size_stride(arg36_1, (480, ), (1, ))
    assert_size_stride(arg37_1, (480, ), (1, ))
    assert_size_stride(arg38_1, (480, ), (1, ))
    assert_size_stride(arg39_1, (480, ), (1, ))
    assert_size_stride(arg40_1, (80, ), (1, ))
    assert_size_stride(arg41_1, (80, ), (1, ))
    assert_size_stride(arg42_1, (480, ), (1, ))
    assert_size_stride(arg43_1, (480, ), (1, ))
    assert_size_stride(arg44_1, (480, ), (1, ))
    assert_size_stride(arg45_1, (480, ), (1, ))
    assert_size_stride(arg46_1, (80, ), (1, ))
    assert_size_stride(arg47_1, (80, ), (1, ))
    assert_size_stride(arg48_1, (480, ), (1, ))
    assert_size_stride(arg49_1, (480, ), (1, ))
    assert_size_stride(arg50_1, (480, ), (1, ))
    assert_size_stride(arg51_1, (480, ), (1, ))
    assert_size_stride(arg52_1, (80, ), (1, ))
    assert_size_stride(arg53_1, (80, ), (1, ))
    assert_size_stride(arg54_1, (480, ), (1, ))
    assert_size_stride(arg55_1, (480, ), (1, ))
    assert_size_stride(arg56_1, (480, ), (1, ))
    assert_size_stride(arg57_1, (480, ), (1, ))
    assert_size_stride(arg58_1, (112, ), (1, ))
    assert_size_stride(arg59_1, (112, ), (1, ))
    assert_size_stride(arg60_1, (672, ), (1, ))
    assert_size_stride(arg61_1, (672, ), (1, ))
    assert_size_stride(arg62_1, (672, ), (1, ))
    assert_size_stride(arg63_1, (672, ), (1, ))
    assert_size_stride(arg64_1, (112, ), (1, ))
    assert_size_stride(arg65_1, (112, ), (1, ))
    assert_size_stride(arg66_1, (672, ), (1, ))
    assert_size_stride(arg67_1, (672, ), (1, ))
    assert_size_stride(arg68_1, (672, ), (1, ))
    assert_size_stride(arg69_1, (672, ), (1, ))
    assert_size_stride(arg70_1, (112, ), (1, ))
    assert_size_stride(arg71_1, (112, ), (1, ))
    assert_size_stride(arg72_1, (672, ), (1, ))
    assert_size_stride(arg73_1, (672, ), (1, ))
    assert_size_stride(arg74_1, (672, ), (1, ))
    assert_size_stride(arg75_1, (672, ), (1, ))
    assert_size_stride(arg76_1, (112, ), (1, ))
    assert_size_stride(arg77_1, (112, ), (1, ))
    assert_size_stride(arg78_1, (672, ), (1, ))
    assert_size_stride(arg79_1, (672, ), (1, ))
    assert_size_stride(arg80_1, (672, ), (1, ))
    assert_size_stride(arg81_1, (672, ), (1, ))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (1152, ), (1, ))
    assert_size_stride(arg85_1, (1152, ), (1, ))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (1152, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (192, ), (1, ))
    assert_size_stride(arg90_1, (1152, ), (1, ))
    assert_size_stride(arg91_1, (1152, ), (1, ))
    assert_size_stride(arg92_1, (1152, ), (1, ))
    assert_size_stride(arg93_1, (1152, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, ), (1, ))
    assert_size_stride(arg96_1, (1152, ), (1, ))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (1152, ), (1, ))
    assert_size_stride(arg99_1, (1152, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, ), (1, ))
    assert_size_stride(arg102_1, (1152, ), (1, ))
    assert_size_stride(arg103_1, (1152, ), (1, ))
    assert_size_stride(arg104_1, (1152, ), (1, ))
    assert_size_stride(arg105_1, (1152, ), (1, ))
    assert_size_stride(arg106_1, (192, ), (1, ))
    assert_size_stride(arg107_1, (192, ), (1, ))
    assert_size_stride(arg108_1, (1152, ), (1, ))
    assert_size_stride(arg109_1, (1152, ), (1, ))
    assert_size_stride(arg110_1, (1152, ), (1, ))
    assert_size_stride(arg111_1, (1152, ), (1, ))
    assert_size_stride(arg112_1, (320, ), (1, ))
    assert_size_stride(arg113_1, (320, ), (1, ))
    assert_size_stride(arg114_1, (1280, ), (1, ))
    assert_size_stride(arg115_1, (1280, ), (1, ))
    assert_size_stride(arg116_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg117_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg119_1, (8, ), (1, ))
    assert_size_stride(arg120_1, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg121_1, (32, ), (1, ))
    assert_size_stride(arg122_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg123_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg124_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg125_1, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg126_1, (4, ), (1, ))
    assert_size_stride(arg127_1, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg128_1, (96, ), (1, ))
    assert_size_stride(arg129_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg130_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg131_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg132_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg133_1, (6, ), (1, ))
    assert_size_stride(arg134_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg135_1, (144, ), (1, ))
    assert_size_stride(arg136_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg137_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg138_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg139_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg140_1, (6, ), (1, ))
    assert_size_stride(arg141_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg142_1, (144, ), (1, ))
    assert_size_stride(arg143_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg144_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg145_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg146_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg147_1, (10, ), (1, ))
    assert_size_stride(arg148_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg149_1, (240, ), (1, ))
    assert_size_stride(arg150_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg151_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg152_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg153_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg154_1, (10, ), (1, ))
    assert_size_stride(arg155_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg156_1, (240, ), (1, ))
    assert_size_stride(arg157_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg158_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg159_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg161_1, (20, ), (1, ))
    assert_size_stride(arg162_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg163_1, (480, ), (1, ))
    assert_size_stride(arg164_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg165_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg166_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg167_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg168_1, (20, ), (1, ))
    assert_size_stride(arg169_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg170_1, (480, ), (1, ))
    assert_size_stride(arg171_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg172_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg173_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg174_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg175_1, (20, ), (1, ))
    assert_size_stride(arg176_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg177_1, (480, ), (1, ))
    assert_size_stride(arg178_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg179_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg180_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg181_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg182_1, (20, ), (1, ))
    assert_size_stride(arg183_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg184_1, (480, ), (1, ))
    assert_size_stride(arg185_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg186_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg187_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg188_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg189_1, (28, ), (1, ))
    assert_size_stride(arg190_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg191_1, (672, ), (1, ))
    assert_size_stride(arg192_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg193_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg194_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg195_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg196_1, (28, ), (1, ))
    assert_size_stride(arg197_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg198_1, (672, ), (1, ))
    assert_size_stride(arg199_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg200_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg201_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg202_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg203_1, (28, ), (1, ))
    assert_size_stride(arg204_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg205_1, (672, ), (1, ))
    assert_size_stride(arg206_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg207_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg208_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg209_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg210_1, (28, ), (1, ))
    assert_size_stride(arg211_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg212_1, (672, ), (1, ))
    assert_size_stride(arg213_1, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg214_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg215_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg216_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg217_1, (48, ), (1, ))
    assert_size_stride(arg218_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg219_1, (1152, ), (1, ))
    assert_size_stride(arg220_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg221_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg222_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg223_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg224_1, (48, ), (1, ))
    assert_size_stride(arg225_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg226_1, (1152, ), (1, ))
    assert_size_stride(arg227_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg228_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg229_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg230_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg231_1, (48, ), (1, ))
    assert_size_stride(arg232_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg233_1, (1152, ), (1, ))
    assert_size_stride(arg234_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg235_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg236_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg237_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg238_1, (48, ), (1, ))
    assert_size_stride(arg239_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg240_1, (1152, ), (1, ))
    assert_size_stride(arg241_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg242_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg243_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg244_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg245_1, (48, ), (1, ))
    assert_size_stride(arg246_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg247_1, (1152, ), (1, ))
    assert_size_stride(arg248_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg249_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg250_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg251_1, (1000, ), (1, ))
    assert_size_stride(arg252_1, (32, ), (1, ))
    assert_size_stride(arg253_1, (32, ), (1, ))
    assert_size_stride(arg254_1, (32, ), (1, ))
    assert_size_stride(arg255_1, (32, ), (1, ))
    assert_size_stride(arg256_1, (16, ), (1, ))
    assert_size_stride(arg257_1, (16, ), (1, ))
    assert_size_stride(arg258_1, (96, ), (1, ))
    assert_size_stride(arg259_1, (96, ), (1, ))
    assert_size_stride(arg260_1, (96, ), (1, ))
    assert_size_stride(arg261_1, (96, ), (1, ))
    assert_size_stride(arg262_1, (24, ), (1, ))
    assert_size_stride(arg263_1, (24, ), (1, ))
    assert_size_stride(arg264_1, (144, ), (1, ))
    assert_size_stride(arg265_1, (144, ), (1, ))
    assert_size_stride(arg266_1, (144, ), (1, ))
    assert_size_stride(arg267_1, (144, ), (1, ))
    assert_size_stride(arg268_1, (24, ), (1, ))
    assert_size_stride(arg269_1, (24, ), (1, ))
    assert_size_stride(arg270_1, (144, ), (1, ))
    assert_size_stride(arg271_1, (144, ), (1, ))
    assert_size_stride(arg272_1, (144, ), (1, ))
    assert_size_stride(arg273_1, (144, ), (1, ))
    assert_size_stride(arg274_1, (40, ), (1, ))
    assert_size_stride(arg275_1, (40, ), (1, ))
    assert_size_stride(arg276_1, (240, ), (1, ))
    assert_size_stride(arg277_1, (240, ), (1, ))
    assert_size_stride(arg278_1, (240, ), (1, ))
    assert_size_stride(arg279_1, (240, ), (1, ))
    assert_size_stride(arg280_1, (40, ), (1, ))
    assert_size_stride(arg281_1, (40, ), (1, ))
    assert_size_stride(arg282_1, (240, ), (1, ))
    assert_size_stride(arg283_1, (240, ), (1, ))
    assert_size_stride(arg284_1, (240, ), (1, ))
    assert_size_stride(arg285_1, (240, ), (1, ))
    assert_size_stride(arg286_1, (80, ), (1, ))
    assert_size_stride(arg287_1, (80, ), (1, ))
    assert_size_stride(arg288_1, (480, ), (1, ))
    assert_size_stride(arg289_1, (480, ), (1, ))
    assert_size_stride(arg290_1, (480, ), (1, ))
    assert_size_stride(arg291_1, (480, ), (1, ))
    assert_size_stride(arg292_1, (80, ), (1, ))
    assert_size_stride(arg293_1, (80, ), (1, ))
    assert_size_stride(arg294_1, (480, ), (1, ))
    assert_size_stride(arg295_1, (480, ), (1, ))
    assert_size_stride(arg296_1, (480, ), (1, ))
    assert_size_stride(arg297_1, (480, ), (1, ))
    assert_size_stride(arg298_1, (80, ), (1, ))
    assert_size_stride(arg299_1, (80, ), (1, ))
    assert_size_stride(arg300_1, (480, ), (1, ))
    assert_size_stride(arg301_1, (480, ), (1, ))
    assert_size_stride(arg302_1, (480, ), (1, ))
    assert_size_stride(arg303_1, (480, ), (1, ))
    assert_size_stride(arg304_1, (80, ), (1, ))
    assert_size_stride(arg305_1, (80, ), (1, ))
    assert_size_stride(arg306_1, (480, ), (1, ))
    assert_size_stride(arg307_1, (480, ), (1, ))
    assert_size_stride(arg308_1, (480, ), (1, ))
    assert_size_stride(arg309_1, (480, ), (1, ))
    assert_size_stride(arg310_1, (112, ), (1, ))
    assert_size_stride(arg311_1, (112, ), (1, ))
    assert_size_stride(arg312_1, (672, ), (1, ))
    assert_size_stride(arg313_1, (672, ), (1, ))
    assert_size_stride(arg314_1, (672, ), (1, ))
    assert_size_stride(arg315_1, (672, ), (1, ))
    assert_size_stride(arg316_1, (112, ), (1, ))
    assert_size_stride(arg317_1, (112, ), (1, ))
    assert_size_stride(arg318_1, (672, ), (1, ))
    assert_size_stride(arg319_1, (672, ), (1, ))
    assert_size_stride(arg320_1, (672, ), (1, ))
    assert_size_stride(arg321_1, (672, ), (1, ))
    assert_size_stride(arg322_1, (112, ), (1, ))
    assert_size_stride(arg323_1, (112, ), (1, ))
    assert_size_stride(arg324_1, (672, ), (1, ))
    assert_size_stride(arg325_1, (672, ), (1, ))
    assert_size_stride(arg326_1, (672, ), (1, ))
    assert_size_stride(arg327_1, (672, ), (1, ))
    assert_size_stride(arg328_1, (112, ), (1, ))
    assert_size_stride(arg329_1, (112, ), (1, ))
    assert_size_stride(arg330_1, (672, ), (1, ))
    assert_size_stride(arg331_1, (672, ), (1, ))
    assert_size_stride(arg332_1, (672, ), (1, ))
    assert_size_stride(arg333_1, (672, ), (1, ))
    assert_size_stride(arg334_1, (192, ), (1, ))
    assert_size_stride(arg335_1, (192, ), (1, ))
    assert_size_stride(arg336_1, (1152, ), (1, ))
    assert_size_stride(arg337_1, (1152, ), (1, ))
    assert_size_stride(arg338_1, (1152, ), (1, ))
    assert_size_stride(arg339_1, (1152, ), (1, ))
    assert_size_stride(arg340_1, (192, ), (1, ))
    assert_size_stride(arg341_1, (192, ), (1, ))
    assert_size_stride(arg342_1, (1152, ), (1, ))
    assert_size_stride(arg343_1, (1152, ), (1, ))
    assert_size_stride(arg344_1, (1152, ), (1, ))
    assert_size_stride(arg345_1, (1152, ), (1, ))
    assert_size_stride(arg346_1, (192, ), (1, ))
    assert_size_stride(arg347_1, (192, ), (1, ))
    assert_size_stride(arg348_1, (1152, ), (1, ))
    assert_size_stride(arg349_1, (1152, ), (1, ))
    assert_size_stride(arg350_1, (1152, ), (1, ))
    assert_size_stride(arg351_1, (1152, ), (1, ))
    assert_size_stride(arg352_1, (192, ), (1, ))
    assert_size_stride(arg353_1, (192, ), (1, ))
    assert_size_stride(arg354_1, (1152, ), (1, ))
    assert_size_stride(arg355_1, (1152, ), (1, ))
    assert_size_stride(arg356_1, (1152, ), (1, ))
    assert_size_stride(arg357_1, (1152, ), (1, ))
    assert_size_stride(arg358_1, (192, ), (1, ))
    assert_size_stride(arg359_1, (192, ), (1, ))
    assert_size_stride(arg360_1, (1152, ), (1, ))
    assert_size_stride(arg361_1, (1152, ), (1, ))
    assert_size_stride(arg362_1, (1152, ), (1, ))
    assert_size_stride(arg363_1, (1152, ), (1, ))
    assert_size_stride(arg364_1, (320, ), (1, ))
    assert_size_stride(arg365_1, (320, ), (1, ))
    assert_size_stride(arg366_1, (1280, ), (1, ))
    assert_size_stride(arg367_1, (1280, ), (1, ))
    assert_size_stride(arg368_1, (8, 3, 192, 192), (110592, 36864, 192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 192, 192), (110592, 1, 576, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg368_1, buf0, 24, 36864, grid=grid(24, 36864), stream=stream0)
        del arg368_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg116_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg116_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 96, 96), (294912, 9216, 96, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 32, 96, 96), (294912, 1, 3072, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg252_1, arg253_1, arg0_1, arg1_1, buf4, 256, 9216, grid=grid(256, 9216), stream=stream0)
        del arg0_1
        del arg1_1
        del arg252_1
        del arg253_1
        del buf3
        # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.silu]
        buf5 = extern_kernels.convolution(buf4, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 96, 96), (294912, 9216, 96, 1))
        del arg117_1
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf8 = reinterpret_tensor(buf7, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf7  # reuse
        # Source Nodes: [x_6, x_9, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3.run(buf6, buf8, arg254_1, arg255_1, arg2_1, arg3_1, 256, 9216, grid=grid(256), stream=stream0)
        del arg254_1
        del arg255_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_9, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf9 = extern_kernels.convolution(buf8, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg118_1
        del buf8
        buf10 = reinterpret_tensor(buf9, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf9  # reuse
        # Source Nodes: [x_9, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_4.run(buf10, arg119_1, 64, grid=grid(64), stream=stream0)
        del arg119_1
        # Source Nodes: [x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf11 = extern_kernels.convolution(buf10, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg120_1
        del buf10
        buf12 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_5.run(buf6, buf11, arg121_1, buf12, 256, 9216, grid=grid(256, 9216), stream=stream0)
        del arg121_1
        del buf11
        del buf6
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf13 = extern_kernels.convolution(buf12, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 16, 96, 96), (147456, 9216, 96, 1))
        del arg122_1
        del buf12
        buf14 = empty_strided((8, 16, 96, 96), (147456, 1, 1536, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf13, arg256_1, arg257_1, arg4_1, arg5_1, buf14, 128, 9216, grid=grid(128, 9216), stream=stream0)
        del arg256_1
        del arg257_1
        del arg4_1
        del arg5_1
        del buf13
        # Source Nodes: [x_12, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 96, 96, 96), (884736, 9216, 96, 1))
        del arg123_1
        del buf14
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided((8, 96, 96, 96), (884736, 1, 9216, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_7.run(buf16, arg258_1, arg259_1, arg6_1, arg7_1, buf17, 768, 9216, grid=grid(768, 9216), stream=stream0)
        del arg258_1
        del arg259_1
        del arg6_1
        del arg7_1
        del buf16
        # Source Nodes: [x_20, x_21], Original ATen: [aten.convolution, aten.silu]
        buf18 = extern_kernels.convolution(buf17, arg124_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf18, (8, 96, 48, 48), (221184, 2304, 48, 1))
        del arg124_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf20, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf20  # reuse
        # Source Nodes: [x_22, x_25, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8.run(buf19, buf21, arg260_1, arg261_1, arg8_1, arg9_1, 768, 2304, grid=grid(768), stream=stream0)
        del arg260_1
        del arg261_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_25, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf22 = extern_kernels.convolution(buf21, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 4, 1, 1), (4, 1, 1, 1))
        del arg125_1
        del buf21
        buf23 = reinterpret_tensor(buf22, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf22  # reuse
        # Source Nodes: [x_25, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_9.run(buf23, arg126_1, 32, grid=grid(32), stream=stream0)
        del arg126_1
        # Source Nodes: [x_25, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf24 = extern_kernels.convolution(buf23, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 96, 1, 1), (96, 1, 1, 1))
        del arg127_1
        del buf23
        buf25 = empty_strided((8, 96, 48, 48), (221184, 1, 4608, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_10.run(buf19, buf24, arg128_1, buf25, 768, 2304, grid=grid(768, 2304), stream=stream0)
        del arg128_1
        del buf19
        del buf24
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_27, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf26 = extern_kernels.convolution(buf25, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 24, 48, 48), (55296, 2304, 48, 1))
        del arg129_1
        del buf25
        buf27 = empty_strided((8, 24, 48, 48), (55296, 1, 1152, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf26, arg262_1, arg263_1, arg10_1, arg11_1, buf27, 192, 2304, grid=grid(192, 2304), stream=stream0)
        del arg10_1
        del arg11_1
        del arg262_1
        del arg263_1
        del buf26
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 144, 48, 48), (331776, 2304, 48, 1))
        del arg130_1
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided((8, 144, 48, 48), (331776, 1, 6912, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf29, arg264_1, arg265_1, arg12_1, arg13_1, buf30, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del arg12_1
        del arg13_1
        del arg264_1
        del arg265_1
        del buf29
        # Source Nodes: [x_36, x_37], Original ATen: [aten.convolution, aten.silu]
        buf31 = extern_kernels.convolution(buf30, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf31, (8, 144, 48, 48), (331776, 2304, 48, 1))
        del arg131_1
        buf32 = buf31; del buf31  # reuse
        buf33 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf34 = reinterpret_tensor(buf33, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf33  # reuse
        # Source Nodes: [x_38, x_41, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_13.run(buf32, buf34, arg266_1, arg267_1, arg14_1, arg15_1, 1152, 2304, grid=grid(1152), stream=stream0)
        del arg14_1
        del arg15_1
        del arg266_1
        del arg267_1
        # Source Nodes: [x_41, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf35 = extern_kernels.convolution(buf34, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg132_1
        del buf34
        buf36 = reinterpret_tensor(buf35, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf35  # reuse
        # Source Nodes: [x_41, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_14.run(buf36, arg133_1, 48, grid=grid(48), stream=stream0)
        del arg133_1
        # Source Nodes: [x_41, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg134_1
        del buf36
        buf38 = buf30; del buf30  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_15.run(buf32, buf37, arg135_1, buf38, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del arg135_1
        del buf32
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_43, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf39 = extern_kernels.convolution(buf38, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 24, 48, 48), (55296, 2304, 48, 1))
        del arg136_1
        buf40 = buf27; del buf27  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf40, buf39, arg268_1, arg269_1, arg16_1, arg17_1, 18432, 24, grid=grid(18432, 24), stream=stream0)
        del arg16_1
        del arg17_1
        del arg268_1
        del arg269_1
        del buf39
        # Source Nodes: [shortcut_3, x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 144, 48, 48), (331776, 2304, 48, 1))
        del arg137_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        buf43 = buf38; del buf38  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_12.run(buf42, arg270_1, arg271_1, arg18_1, arg19_1, buf43, 1152, 2304, grid=grid(1152, 2304), stream=stream0)
        del arg18_1
        del arg19_1
        del arg270_1
        del arg271_1
        del buf42
        # Source Nodes: [x_53, x_54], Original ATen: [aten.convolution, aten.silu]
        buf44 = extern_kernels.convolution(buf43, arg138_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf44, (8, 144, 24, 24), (82944, 576, 24, 1))
        del arg138_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        buf46 = reinterpret_tensor(buf37, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf37  # reuse
        buf47 = reinterpret_tensor(buf46, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf46  # reuse
        # Source Nodes: [x_55, x_58, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_17.run(buf45, buf47, arg272_1, arg273_1, arg20_1, arg21_1, 1152, 576, grid=grid(1152), stream=stream0)
        del arg20_1
        del arg21_1
        del arg272_1
        del arg273_1
        # Source Nodes: [x_58, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf48 = extern_kernels.convolution(buf47, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg139_1
        del buf47
        buf49 = reinterpret_tensor(buf48, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf48  # reuse
        # Source Nodes: [x_58, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_14.run(buf49, arg140_1, 48, grid=grid(48), stream=stream0)
        del arg140_1
        # Source Nodes: [x_58, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf50 = extern_kernels.convolution(buf49, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg141_1
        del buf49
        buf51 = empty_strided((8, 144, 24, 24), (82944, 1, 3456, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_18.run(buf45, buf50, arg142_1, buf51, 1152, 576, grid=grid(1152, 576), stream=stream0)
        del arg142_1
        del buf45
        del buf50
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_60, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf52 = extern_kernels.convolution(buf51, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 40, 24, 24), (23040, 576, 24, 1))
        del arg143_1
        del buf51
        buf53 = empty_strided((8, 40, 24, 24), (23040, 1, 960, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf52, arg274_1, arg275_1, arg22_1, arg23_1, buf53, 320, 576, grid=grid(320, 576), stream=stream0)
        del arg22_1
        del arg23_1
        del arg274_1
        del arg275_1
        del buf52
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 240, 24, 24), (138240, 576, 24, 1))
        del arg144_1
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided((8, 240, 24, 24), (138240, 1, 5760, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf55, arg276_1, arg277_1, arg24_1, arg25_1, buf56, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del arg24_1
        del arg25_1
        del arg276_1
        del arg277_1
        del buf55
        # Source Nodes: [x_69, x_70], Original ATen: [aten.convolution, aten.silu]
        buf57 = extern_kernels.convolution(buf56, arg145_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf57, (8, 240, 24, 24), (138240, 576, 24, 1))
        del arg145_1
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf60 = reinterpret_tensor(buf59, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf59  # reuse
        # Source Nodes: [x_71, x_74, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_21.run(buf58, buf60, arg278_1, arg279_1, arg26_1, arg27_1, 1920, 576, grid=grid(1920), stream=stream0)
        del arg26_1
        del arg278_1
        del arg279_1
        del arg27_1
        # Source Nodes: [x_74, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf61 = extern_kernels.convolution(buf60, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg146_1
        del buf60
        buf62 = reinterpret_tensor(buf61, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf61  # reuse
        # Source Nodes: [x_74, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_22.run(buf62, arg147_1, 80, grid=grid(80), stream=stream0)
        del arg147_1
        # Source Nodes: [x_74, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf63 = extern_kernels.convolution(buf62, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg148_1
        del buf62
        buf64 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_23.run(buf58, buf63, arg149_1, buf64, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del arg149_1
        del buf58
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_76, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf65 = extern_kernels.convolution(buf64, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 40, 24, 24), (23040, 576, 24, 1))
        del arg150_1
        buf66 = buf53; del buf53  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf66, buf65, arg280_1, arg281_1, arg28_1, arg29_1, 4608, 40, grid=grid(4608, 40), stream=stream0)
        del arg280_1
        del arg281_1
        del arg28_1
        del arg29_1
        del buf65
        # Source Nodes: [shortcut_5, x_77, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 240, 24, 24), (138240, 576, 24, 1))
        del arg151_1
        del buf66
        buf68 = buf67; del buf67  # reuse
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_20.run(buf68, arg282_1, arg283_1, arg30_1, arg31_1, buf69, 1920, 576, grid=grid(1920, 576), stream=stream0)
        del arg282_1
        del arg283_1
        del arg30_1
        del arg31_1
        del buf68
        # Source Nodes: [x_86, x_87], Original ATen: [aten.convolution, aten.silu]
        buf70 = extern_kernels.convolution(buf69, arg152_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf70, (8, 240, 12, 12), (34560, 144, 12, 1))
        del arg152_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        buf72 = reinterpret_tensor(buf63, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf63  # reuse
        buf73 = reinterpret_tensor(buf72, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf72  # reuse
        # Source Nodes: [x_88, x_91, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_25.run(buf71, buf73, arg284_1, arg285_1, arg32_1, arg33_1, 1920, 144, grid=grid(1920), stream=stream0)
        del arg284_1
        del arg285_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_91, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf74 = extern_kernels.convolution(buf73, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg153_1
        del buf73
        buf75 = reinterpret_tensor(buf74, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf74  # reuse
        # Source Nodes: [x_91, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_22.run(buf75, arg154_1, 80, grid=grid(80), stream=stream0)
        del arg154_1
        # Source Nodes: [x_91, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf76 = extern_kernels.convolution(buf75, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg155_1
        del buf75
        buf77 = empty_strided((8, 240, 12, 12), (34560, 1, 2880, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_26.run(buf71, buf76, arg156_1, buf77, 1920, 144, grid=grid(1920, 144), stream=stream0)
        del arg156_1
        del buf71
        del buf76
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_93, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf78 = extern_kernels.convolution(buf77, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 80, 12, 12), (11520, 144, 12, 1))
        del arg157_1
        del buf77
        buf79 = empty_strided((8, 80, 12, 12), (11520, 1, 960, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf78, arg286_1, arg287_1, arg34_1, arg35_1, buf79, 640, 144, grid=grid(640, 144), stream=stream0)
        del arg286_1
        del arg287_1
        del arg34_1
        del arg35_1
        del buf78
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg158_1
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided((8, 480, 12, 12), (69120, 1, 5760, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf81, arg288_1, arg289_1, arg36_1, arg37_1, buf82, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg288_1
        del arg289_1
        del arg36_1
        del arg37_1
        del buf81
        # Source Nodes: [x_102, x_103], Original ATen: [aten.convolution, aten.silu]
        buf83 = extern_kernels.convolution(buf82, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf83, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg159_1
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf86 = reinterpret_tensor(buf85, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf85  # reuse
        # Source Nodes: [x_104, x_107, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf84, buf86, arg290_1, arg291_1, arg38_1, arg39_1, 3840, 144, grid=grid(3840), stream=stream0)
        del arg290_1
        del arg291_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_107, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf87 = extern_kernels.convolution(buf86, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg160_1
        del buf86
        buf88 = reinterpret_tensor(buf87, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf87  # reuse
        # Source Nodes: [x_107, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_30.run(buf88, arg161_1, 160, grid=grid(160), stream=stream0)
        del arg161_1
        # Source Nodes: [x_107, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf89 = extern_kernels.convolution(buf88, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg162_1
        del buf88
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_31.run(buf84, buf89, arg163_1, buf90, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg163_1
        del buf84
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_109, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf91 = extern_kernels.convolution(buf90, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 80, 12, 12), (11520, 144, 12, 1))
        del arg164_1
        buf92 = buf79; del buf79  # reuse
        # Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf92, buf91, arg292_1, arg293_1, arg40_1, arg41_1, 1152, 80, grid=grid(1152, 80), stream=stream0)
        del arg292_1
        del arg293_1
        del arg40_1
        del arg41_1
        del buf91
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg165_1
        buf94 = buf93; del buf93  # reuse
        buf95 = buf90; del buf90  # reuse
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf94, arg294_1, arg295_1, arg42_1, arg43_1, buf95, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg294_1
        del arg295_1
        del arg42_1
        del arg43_1
        del buf94
        # Source Nodes: [x_119, x_120], Original ATen: [aten.convolution, aten.silu]
        buf96 = extern_kernels.convolution(buf95, arg166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf96, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg166_1
        buf97 = buf96; del buf96  # reuse
        buf98 = reinterpret_tensor(buf89, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf89  # reuse
        buf99 = reinterpret_tensor(buf98, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf98  # reuse
        # Source Nodes: [x_121, x_124, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf97, buf99, arg296_1, arg297_1, arg44_1, arg45_1, 3840, 144, grid=grid(3840), stream=stream0)
        del arg296_1
        del arg297_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_124, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf100 = extern_kernels.convolution(buf99, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg167_1
        del buf99
        buf101 = reinterpret_tensor(buf100, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf100  # reuse
        # Source Nodes: [x_124, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_30.run(buf101, arg168_1, 160, grid=grid(160), stream=stream0)
        del arg168_1
        # Source Nodes: [x_124, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf102 = extern_kernels.convolution(buf101, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg169_1
        del buf101
        buf103 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_31.run(buf97, buf102, arg170_1, buf103, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg170_1
        del buf97
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125, x_126, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf104 = extern_kernels.convolution(buf103, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 80, 12, 12), (11520, 144, 12, 1))
        del arg171_1
        buf105 = buf92; del buf92  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf105, buf104, arg298_1, arg299_1, arg46_1, arg47_1, 1152, 80, grid=grid(1152, 80), stream=stream0)
        del arg298_1
        del arg299_1
        del arg46_1
        del arg47_1
        del buf104
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg172_1
        buf107 = buf106; del buf106  # reuse
        buf108 = buf103; del buf103  # reuse
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf107, arg300_1, arg301_1, arg48_1, arg49_1, buf108, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg300_1
        del arg301_1
        del arg48_1
        del arg49_1
        del buf107
        # Source Nodes: [x_136, x_137], Original ATen: [aten.convolution, aten.silu]
        buf109 = extern_kernels.convolution(buf108, arg173_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf109, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg173_1
        buf110 = buf109; del buf109  # reuse
        buf111 = reinterpret_tensor(buf102, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf102  # reuse
        buf112 = reinterpret_tensor(buf111, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf111  # reuse
        # Source Nodes: [x_138, x_141, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf110, buf112, arg302_1, arg303_1, arg50_1, arg51_1, 3840, 144, grid=grid(3840), stream=stream0)
        del arg302_1
        del arg303_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_141, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf113 = extern_kernels.convolution(buf112, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg174_1
        del buf112
        buf114 = reinterpret_tensor(buf113, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf113  # reuse
        # Source Nodes: [x_141, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_30.run(buf114, arg175_1, 160, grid=grid(160), stream=stream0)
        del arg175_1
        # Source Nodes: [x_141, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf115 = extern_kernels.convolution(buf114, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg176_1
        del buf114
        buf116 = buf108; del buf108  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_141, x_142, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_31.run(buf110, buf115, arg177_1, buf116, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg177_1
        del buf110
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_141, x_142, x_143, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf117 = extern_kernels.convolution(buf116, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 80, 12, 12), (11520, 144, 12, 1))
        del arg178_1
        buf118 = buf105; del buf105  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_32.run(buf118, buf117, arg304_1, arg305_1, arg52_1, arg53_1, 1152, 80, grid=grid(1152, 80), stream=stream0)
        del arg304_1
        del arg305_1
        del arg52_1
        del arg53_1
        del buf117
        # Source Nodes: [shortcut_9, x_144, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf119 = extern_kernels.convolution(buf118, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg179_1
        buf120 = buf119; del buf119  # reuse
        buf121 = buf116; del buf116  # reuse
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf120, arg306_1, arg307_1, arg54_1, arg55_1, buf121, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg306_1
        del arg307_1
        del arg54_1
        del arg55_1
        del buf120
        # Source Nodes: [x_153, x_154], Original ATen: [aten.convolution, aten.silu]
        buf122 = extern_kernels.convolution(buf121, arg180_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf122, (8, 480, 12, 12), (69120, 144, 12, 1))
        del arg180_1
        buf123 = buf122; del buf122  # reuse
        buf124 = reinterpret_tensor(buf115, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf115  # reuse
        buf125 = reinterpret_tensor(buf124, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf124  # reuse
        # Source Nodes: [x_155, x_158, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_29.run(buf123, buf125, arg308_1, arg309_1, arg56_1, arg57_1, 3840, 144, grid=grid(3840), stream=stream0)
        del arg308_1
        del arg309_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_158, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf126 = extern_kernels.convolution(buf125, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg181_1
        del buf125
        buf127 = reinterpret_tensor(buf126, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf126  # reuse
        # Source Nodes: [x_158, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_30.run(buf127, arg182_1, 160, grid=grid(160), stream=stream0)
        del arg182_1
        # Source Nodes: [x_158, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf128 = extern_kernels.convolution(buf127, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg183_1
        del buf127
        buf129 = buf121; del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158, x_159, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_31.run(buf123, buf128, arg184_1, buf129, 3840, 144, grid=grid(3840, 144), stream=stream0)
        del arg184_1
        del buf123
        del buf128
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_158, x_159, x_160, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf130 = extern_kernels.convolution(buf129, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 112, 12, 12), (16128, 144, 12, 1))
        del arg185_1
        del buf129
        buf131 = empty_strided((8, 112, 12, 12), (16128, 1, 1344, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf130, arg310_1, arg311_1, arg58_1, arg59_1, buf131, 896, 144, grid=grid(896, 144), stream=stream0)
        del arg310_1
        del arg311_1
        del arg58_1
        del arg59_1
        del buf130
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg186_1
        buf133 = buf132; del buf132  # reuse
        buf134 = empty_strided((8, 672, 12, 12), (96768, 1, 8064, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf133, arg312_1, arg313_1, arg60_1, arg61_1, buf134, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg312_1
        del arg313_1
        del arg60_1
        del arg61_1
        del buf133
        # Source Nodes: [x_169, x_170], Original ATen: [aten.convolution, aten.silu]
        buf135 = extern_kernels.convolution(buf134, arg187_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf135, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg187_1
        buf136 = buf135; del buf135  # reuse
        buf137 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf138 = reinterpret_tensor(buf137, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf137  # reuse
        # Source Nodes: [x_171, x_174, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf136, buf138, arg314_1, arg315_1, arg62_1, arg63_1, 5376, 144, grid=grid(5376), stream=stream0)
        del arg314_1
        del arg315_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_174, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf139 = extern_kernels.convolution(buf138, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg188_1
        del buf138
        buf140 = reinterpret_tensor(buf139, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf139  # reuse
        # Source Nodes: [x_174, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf140, arg189_1, 224, grid=grid(224), stream=stream0)
        del arg189_1
        # Source Nodes: [x_174, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf141 = extern_kernels.convolution(buf140, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg190_1
        del buf140
        buf142 = buf134; del buf134  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf136, buf141, arg191_1, buf142, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg191_1
        del buf136
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_174, x_175, x_176, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf143 = extern_kernels.convolution(buf142, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 112, 12, 12), (16128, 144, 12, 1))
        del arg192_1
        buf144 = buf131; del buf131  # reuse
        # Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf144, buf143, arg316_1, arg317_1, arg64_1, arg65_1, 1152, 112, grid=grid(1152, 112), stream=stream0)
        del arg316_1
        del arg317_1
        del arg64_1
        del arg65_1
        del buf143
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg193_1
        buf146 = buf145; del buf145  # reuse
        buf147 = buf142; del buf142  # reuse
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf146, arg318_1, arg319_1, arg66_1, arg67_1, buf147, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg318_1
        del arg319_1
        del arg66_1
        del arg67_1
        del buf146
        # Source Nodes: [x_186, x_187], Original ATen: [aten.convolution, aten.silu]
        buf148 = extern_kernels.convolution(buf147, arg194_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf148, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg194_1
        buf149 = buf148; del buf148  # reuse
        buf150 = reinterpret_tensor(buf141, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf141  # reuse
        buf151 = reinterpret_tensor(buf150, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf150  # reuse
        # Source Nodes: [x_188, x_191, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf149, buf151, arg320_1, arg321_1, arg68_1, arg69_1, 5376, 144, grid=grid(5376), stream=stream0)
        del arg320_1
        del arg321_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_191, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf152 = extern_kernels.convolution(buf151, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg195_1
        del buf151
        buf153 = reinterpret_tensor(buf152, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf152  # reuse
        # Source Nodes: [x_191, x_se_44, x_se_45, x_se_46], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf153, arg196_1, 224, grid=grid(224), stream=stream0)
        del arg196_1
        # Source Nodes: [x_191, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf154 = extern_kernels.convolution(buf153, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg197_1
        del buf153
        buf155 = buf147; del buf147  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_191, x_192, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf149, buf154, arg198_1, buf155, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg198_1
        del buf149
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_191, x_192, x_193, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf156 = extern_kernels.convolution(buf155, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 112, 12, 12), (16128, 144, 12, 1))
        del arg199_1
        buf157 = buf144; del buf144  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf157, buf156, arg322_1, arg323_1, arg70_1, arg71_1, 1152, 112, grid=grid(1152, 112), stream=stream0)
        del arg322_1
        del arg323_1
        del arg70_1
        del arg71_1
        del buf156
        # Source Nodes: [x_199], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg200_1
        buf159 = buf158; del buf158  # reuse
        buf160 = buf155; del buf155  # reuse
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf159, arg324_1, arg325_1, arg72_1, arg73_1, buf160, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg324_1
        del arg325_1
        del arg72_1
        del arg73_1
        del buf159
        # Source Nodes: [x_203, x_204], Original ATen: [aten.convolution, aten.silu]
        buf161 = extern_kernels.convolution(buf160, arg201_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf161, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg201_1
        buf162 = buf161; del buf161  # reuse
        buf163 = reinterpret_tensor(buf154, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf154  # reuse
        buf164 = reinterpret_tensor(buf163, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf163  # reuse
        # Source Nodes: [x_205, x_208, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_35.run(buf162, buf164, arg326_1, arg327_1, arg74_1, arg75_1, 5376, 144, grid=grid(5376), stream=stream0)
        del arg326_1
        del arg327_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_208, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf165 = extern_kernels.convolution(buf164, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg202_1
        del buf164
        buf166 = reinterpret_tensor(buf165, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf165  # reuse
        # Source Nodes: [x_208, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf166, arg203_1, 224, grid=grid(224), stream=stream0)
        del arg203_1
        # Source Nodes: [x_208, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf167 = extern_kernels.convolution(buf166, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg204_1
        del buf166
        buf168 = buf160; del buf160  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208, x_209, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_37.run(buf162, buf167, arg205_1, buf168, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg205_1
        del buf162
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_208, x_209, x_210, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf169 = extern_kernels.convolution(buf168, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 112, 12, 12), (16128, 144, 12, 1))
        del arg206_1
        buf170 = buf157; del buf157  # reuse
        # Source Nodes: [shortcut_13, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_38.run(buf170, buf169, arg328_1, arg329_1, arg76_1, arg77_1, 1152, 112, grid=grid(1152, 112), stream=stream0)
        del arg328_1
        del arg329_1
        del arg76_1
        del arg77_1
        del buf169
        # Source Nodes: [shortcut_13, x_211, x_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 672, 12, 12), (96768, 144, 12, 1))
        del arg207_1
        del buf170
        buf172 = buf171; del buf171  # reuse
        buf173 = buf168; del buf168  # reuse
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_34.run(buf172, arg330_1, arg331_1, arg78_1, arg79_1, buf173, 5376, 144, grid=grid(5376, 144), stream=stream0)
        del arg330_1
        del arg331_1
        del arg78_1
        del arg79_1
        del buf172
        # Source Nodes: [x_220, x_221], Original ATen: [aten.convolution, aten.silu]
        buf174 = extern_kernels.convolution(buf173, arg208_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf174, (8, 672, 6, 6), (24192, 36, 6, 1))
        del arg208_1
        del buf173
        buf175 = buf174; del buf174  # reuse
        buf176 = reinterpret_tensor(buf167, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf167  # reuse
        buf177 = reinterpret_tensor(buf176, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf176  # reuse
        # Source Nodes: [x_222, x_225, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_39.run(buf175, buf177, arg332_1, arg333_1, arg80_1, arg81_1, 5376, 36, grid=grid(5376), stream=stream0)
        del arg332_1
        del arg333_1
        del arg80_1
        del arg81_1
        # Source Nodes: [x_225, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf178 = extern_kernels.convolution(buf177, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg209_1
        del buf177
        buf179 = reinterpret_tensor(buf178, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf178  # reuse
        # Source Nodes: [x_225, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_36.run(buf179, arg210_1, 224, grid=grid(224), stream=stream0)
        del arg210_1
        # Source Nodes: [x_225, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf180 = extern_kernels.convolution(buf179, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg211_1
        del buf179
        buf181 = empty_strided((8, 672, 6, 6), (24192, 1, 4032, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_40.run(buf175, buf180, arg212_1, buf181, 5376, 36, grid=grid(5376, 36), stream=stream0)
        del arg212_1
        del buf175
        del buf180
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_225, x_226, x_227, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf182 = extern_kernels.convolution(buf181, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 192, 6, 6), (6912, 36, 6, 1))
        del arg213_1
        del buf181
        buf183 = empty_strided((8, 192, 6, 6), (6912, 1, 1152, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf182, arg334_1, arg335_1, arg82_1, arg83_1, buf183, 1536, 36, grid=grid(1536, 36), stream=stream0)
        del arg334_1
        del arg335_1
        del arg82_1
        del arg83_1
        del buf182
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg214_1
        buf185 = buf184; del buf184  # reuse
        buf186 = empty_strided((8, 1152, 6, 6), (41472, 1, 6912, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf185, arg336_1, arg337_1, arg84_1, arg85_1, buf186, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg336_1
        del arg337_1
        del arg84_1
        del arg85_1
        del buf185
        # Source Nodes: [x_236, x_237], Original ATen: [aten.convolution, aten.silu]
        buf187 = extern_kernels.convolution(buf186, arg215_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf187, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg215_1
        buf188 = buf187; del buf187  # reuse
        buf189 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf190 = reinterpret_tensor(buf189, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf189  # reuse
        # Source Nodes: [x_238, x_241, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43.run(buf188, buf190, arg338_1, arg339_1, arg86_1, arg87_1, 9216, 36, grid=grid(9216), stream=stream0)
        del arg338_1
        del arg339_1
        del arg86_1
        del arg87_1
        # Source Nodes: [x_241, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf191 = extern_kernels.convolution(buf190, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg216_1
        del buf190
        buf192 = reinterpret_tensor(buf191, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf191  # reuse
        # Source Nodes: [x_241, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_44.run(buf192, arg217_1, 384, grid=grid(384), stream=stream0)
        del arg217_1
        # Source Nodes: [x_241, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf193 = extern_kernels.convolution(buf192, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg218_1
        del buf192
        buf194 = buf186; del buf186  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf188, buf193, arg219_1, buf194, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg219_1
        del buf188
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_241, x_242, x_243, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf195 = extern_kernels.convolution(buf194, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 192, 6, 6), (6912, 36, 6, 1))
        del arg220_1
        buf196 = buf183; del buf183  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf196, buf195, arg340_1, arg341_1, arg88_1, arg89_1, 288, 192, grid=grid(288, 192), stream=stream0)
        del arg340_1
        del arg341_1
        del arg88_1
        del arg89_1
        del buf195
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg221_1
        buf198 = buf197; del buf197  # reuse
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf198, arg342_1, arg343_1, arg90_1, arg91_1, buf199, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg342_1
        del arg343_1
        del arg90_1
        del arg91_1
        del buf198
        # Source Nodes: [x_253, x_254], Original ATen: [aten.convolution, aten.silu]
        buf200 = extern_kernels.convolution(buf199, arg222_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf200, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg222_1
        buf201 = buf200; del buf200  # reuse
        buf202 = reinterpret_tensor(buf193, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf193  # reuse
        buf203 = reinterpret_tensor(buf202, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf202  # reuse
        # Source Nodes: [x_255, x_258, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43.run(buf201, buf203, arg344_1, arg345_1, arg92_1, arg93_1, 9216, 36, grid=grid(9216), stream=stream0)
        del arg344_1
        del arg345_1
        del arg92_1
        del arg93_1
        # Source Nodes: [x_258, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf204 = extern_kernels.convolution(buf203, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg223_1
        del buf203
        buf205 = reinterpret_tensor(buf204, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf204  # reuse
        # Source Nodes: [x_258, x_se_60, x_se_61, x_se_62], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_44.run(buf205, arg224_1, 384, grid=grid(384), stream=stream0)
        del arg224_1
        # Source Nodes: [x_258, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf206 = extern_kernels.convolution(buf205, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg225_1
        del buf205
        buf207 = buf199; del buf199  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_258, x_259, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf201, buf206, arg226_1, buf207, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg226_1
        del buf201
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_258, x_259, x_260, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf208 = extern_kernels.convolution(buf207, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 192, 6, 6), (6912, 36, 6, 1))
        del arg227_1
        buf209 = buf196; del buf196  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf209, buf208, arg346_1, arg347_1, arg94_1, arg95_1, 288, 192, grid=grid(288, 192), stream=stream0)
        del arg346_1
        del arg347_1
        del arg94_1
        del arg95_1
        del buf208
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg228_1
        buf211 = buf210; del buf210  # reuse
        buf212 = buf207; del buf207  # reuse
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf211, arg348_1, arg349_1, arg96_1, arg97_1, buf212, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg348_1
        del arg349_1
        del arg96_1
        del arg97_1
        del buf211
        # Source Nodes: [x_270, x_271], Original ATen: [aten.convolution, aten.silu]
        buf213 = extern_kernels.convolution(buf212, arg229_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf213, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg229_1
        buf214 = buf213; del buf213  # reuse
        buf215 = reinterpret_tensor(buf206, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf206  # reuse
        buf216 = reinterpret_tensor(buf215, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf215  # reuse
        # Source Nodes: [x_272, x_275, x_se_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43.run(buf214, buf216, arg350_1, arg351_1, arg98_1, arg99_1, 9216, 36, grid=grid(9216), stream=stream0)
        del arg350_1
        del arg351_1
        del arg98_1
        del arg99_1
        # Source Nodes: [x_275, x_se_64, x_se_65], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf217 = extern_kernels.convolution(buf216, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg230_1
        del buf216
        buf218 = reinterpret_tensor(buf217, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf217  # reuse
        # Source Nodes: [x_275, x_se_64, x_se_65, x_se_66], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_44.run(buf218, arg231_1, 384, grid=grid(384), stream=stream0)
        del arg231_1
        # Source Nodes: [x_275, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf219 = extern_kernels.convolution(buf218, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg232_1
        del buf218
        buf220 = buf212; del buf212  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_275, x_276, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf214, buf219, arg233_1, buf220, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg233_1
        del buf214
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_275, x_276, x_277, x_se_64, x_se_65, x_se_66, x_se_67], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf221 = extern_kernels.convolution(buf220, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 192, 6, 6), (6912, 36, 6, 1))
        del arg234_1
        buf222 = buf209; del buf209  # reuse
        # Source Nodes: [shortcut_17, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf222, buf221, arg352_1, arg353_1, arg100_1, arg101_1, 288, 192, grid=grid(288, 192), stream=stream0)
        del arg100_1
        del arg101_1
        del arg352_1
        del arg353_1
        del buf221
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg235_1
        buf224 = buf223; del buf223  # reuse
        buf225 = buf220; del buf220  # reuse
        # Source Nodes: [x_284, x_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf224, arg354_1, arg355_1, arg102_1, arg103_1, buf225, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg102_1
        del arg103_1
        del arg354_1
        del arg355_1
        del buf224
        # Source Nodes: [x_287, x_288], Original ATen: [aten.convolution, aten.silu]
        buf226 = extern_kernels.convolution(buf225, arg236_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf226, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg236_1
        buf227 = buf226; del buf226  # reuse
        buf228 = reinterpret_tensor(buf219, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf219  # reuse
        buf229 = reinterpret_tensor(buf228, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf228  # reuse
        # Source Nodes: [x_289, x_292, x_se_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43.run(buf227, buf229, arg356_1, arg357_1, arg104_1, arg105_1, 9216, 36, grid=grid(9216), stream=stream0)
        del arg104_1
        del arg105_1
        del arg356_1
        del arg357_1
        # Source Nodes: [x_292, x_se_68, x_se_69], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf230 = extern_kernels.convolution(buf229, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg237_1
        del buf229
        buf231 = reinterpret_tensor(buf230, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf230  # reuse
        # Source Nodes: [x_292, x_se_68, x_se_69, x_se_70], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_44.run(buf231, arg238_1, 384, grid=grid(384), stream=stream0)
        del arg238_1
        # Source Nodes: [x_292, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf232 = extern_kernels.convolution(buf231, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg239_1
        del buf231
        buf233 = buf225; del buf225  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_292, x_293, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf227, buf232, arg240_1, buf233, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg240_1
        del buf227
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____4___se_gate, x_292, x_293, x_294, x_se_68, x_se_69, x_se_70, x_se_71], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf234 = extern_kernels.convolution(buf233, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 192, 6, 6), (6912, 36, 6, 1))
        del arg241_1
        buf235 = buf222; del buf222  # reuse
        # Source Nodes: [shortcut_18, x_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_46.run(buf235, buf234, arg358_1, arg359_1, arg106_1, arg107_1, 288, 192, grid=grid(288, 192), stream=stream0)
        del arg106_1
        del arg107_1
        del arg358_1
        del arg359_1
        del buf234
        # Source Nodes: [shortcut_18, x_295, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf236 = extern_kernels.convolution(buf235, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg242_1
        del buf235
        buf237 = buf236; del buf236  # reuse
        buf238 = buf233; del buf233  # reuse
        # Source Nodes: [x_301, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_42.run(buf237, arg360_1, arg361_1, arg108_1, arg109_1, buf238, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg108_1
        del arg109_1
        del arg360_1
        del arg361_1
        del buf237
        # Source Nodes: [x_304, x_305], Original ATen: [aten.convolution, aten.silu]
        buf239 = extern_kernels.convolution(buf238, arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf239, (8, 1152, 6, 6), (41472, 36, 6, 1))
        del arg243_1
        buf240 = buf239; del buf239  # reuse
        buf241 = reinterpret_tensor(buf232, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf232  # reuse
        buf242 = reinterpret_tensor(buf241, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf241  # reuse
        # Source Nodes: [x_306, x_309, x_se_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_43.run(buf240, buf242, arg362_1, arg363_1, arg110_1, arg111_1, 9216, 36, grid=grid(9216), stream=stream0)
        del arg110_1
        del arg111_1
        del arg362_1
        del arg363_1
        # Source Nodes: [x_309, x_se_72, x_se_73], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf243 = extern_kernels.convolution(buf242, arg244_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg244_1
        del buf242
        buf244 = reinterpret_tensor(buf243, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf243  # reuse
        # Source Nodes: [x_309, x_se_72, x_se_73, x_se_74], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_44.run(buf244, arg245_1, 384, grid=grid(384), stream=stream0)
        del arg245_1
        # Source Nodes: [x_309, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf245 = extern_kernels.convolution(buf244, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg246_1
        del buf244
        buf246 = buf238; del buf238  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309, x_310, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_45.run(buf240, buf245, arg247_1, buf246, 9216, 36, grid=grid(9216, 36), stream=stream0)
        del arg247_1
        del buf240
        del buf245
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_309, x_310, x_311, x_se_72, x_se_73, x_se_74, x_se_75], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf247 = extern_kernels.convolution(buf246, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 320, 6, 6), (11520, 36, 6, 1))
        del arg248_1
        del buf246
        buf248 = reinterpret_tensor(buf118, (8, 320, 6, 6), (11520, 1, 1920, 320), 0); del buf118  # reuse
        # Source Nodes: [x_312], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_47.run(buf247, arg364_1, arg365_1, arg112_1, arg113_1, buf248, 2560, 36, grid=grid(2560, 36), stream=stream0)
        del arg112_1
        del arg113_1
        del arg364_1
        del arg365_1
        del buf247
        # Source Nodes: [x_312, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf249 = extern_kernels.convolution(buf248, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 1280, 6, 6), (46080, 36, 6, 1))
        del arg249_1
        del buf248
        buf250 = buf249; del buf249  # reuse
        buf251 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf252 = reinterpret_tensor(buf251, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf251  # reuse
        # Source Nodes: [x_318, x_322, x_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_48.run(buf250, buf252, arg366_1, arg367_1, arg114_1, arg115_1, 10240, 36, grid=grid(10240), stream=stream0)
        del arg114_1
        del arg115_1
        del arg366_1
        del arg367_1
        del buf250
        buf253 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg251_1, reinterpret_tensor(buf252, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg250_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf253)
        del arg250_1
        del arg251_1
        return (buf253, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((8, 3, 192, 192), (110592, 36864, 192, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tinynet_a', benchmark_compiled_module)
