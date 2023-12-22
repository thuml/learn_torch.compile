
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


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fjmeqaknlpk5m5fqj6h6x72sutinrgbi6evb72wghecwgcpgd3.py
# Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50625
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 225)
    x2 = xindex % 225
    y4 = yindex
    x5 = xindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 224, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (224*x3) + (50176*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (y0 + (3*x5) + (151875*y1)), tmp8, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/2s/c2su6u62tlywk2wwcmiaqmeerposmvshczsj5okff2heff2im3si.py
# Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
# x => constant_pad_nd
# x_1 => convolution
triton_poi_fused_constant_pad_nd_convolution_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jn/cjn5azjye7l6zr5kz6v7rnphdsa3zwiigfvf3begx3mxqeznbyc7.py
# Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# shortcut => mul_3, sigmoid
# x_2 => add_1, mul_1, mul_2, sub
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
    xnumel = 12544
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g7/cg7xsg46qdqfqxfhjkc5cz33j4pzd26itkxv3rg6zczulq7vmpqn.py
# Source Nodes: [x_10, x_7, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_10 => mul_7, sigmoid_1
# x_7 => add_3, mul_5, mul_6, sub_1
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
    rnumel = 12544
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
        tmp0 = tl.load(in_out_ptr0 + (r2 + (12544*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 0.001
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
        tl.store(in_out_ptr0 + (r2 + (12544*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 12544.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2n/c2nspemircglvj5s24b47x4mudxqjfk7c2thagb5v6qb5c47gxe3.py
# Source Nodes: [x_10, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_10 => mul_7, sigmoid_1
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


# kernel path: /tmp/torchinductor_youkaichao/kn/ckn7wni6uel5upoghqvau3njoa2nlqlwh2w4fm6nc7pkzlcsgqkw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_7, sigmoid_1
# x_11 => mul_9
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtdqcfe5cwbzpn6uyiphsttsshfq3eigtb3ohpfyg4h4xklugnk.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_13 => add_5, mul_11, mul_12, sub_2
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
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7aivgr5j7zjfm34jkkt63tzdntm2qfuetoppsw3kvqv22lufnxa.py
# Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_18 => add_7, mul_14, mul_15, sub_3
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 96
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6eprc4jebkjyqe2fvpn3djwz5pvrba4hr3rh7tknu2kz4t5nwo2.py
# Source Nodes: [x_21, x_23], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_21 => mul_16, sigmoid_4
# x_23 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_silu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12769
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 113)
    x2 = xindex % 113
    y4 = yindex
    x5 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (112*x3) + (12544*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (y0 + (96*x5) + (1225824*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciub6n3b4g2xkvhjnaaghcpa2fwojtzgg4swlzuymkcctvsuq24n.py
# Source Nodes: [x_25, x_28, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_25 => add_9, mul_18, mul_19, sub_4
# x_28 => mul_20, sigmoid_5
# x_se_4 => mean_1
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 3136
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
        tmp0 = tl.load(in_out_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 0.001
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
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 3136.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu43flh44ptlfazagooywccnnkrw6q7eqslzvse2slzia2n7dhu3.py
# Source Nodes: [x_28, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_28 => mul_20, sigmoid_5
# x_se_4 => mean_1
# x_se_5 => convolution_7
# x_se_6 => mul_21, sigmoid_6
triton_poi_fused_convolution_mean_silu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_10', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/iv/civiu4cuugwv2szbqaar3rme3i7c4uwckmcysmok32quyw7m4zs2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28, x_29, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_28 => mul_20, sigmoid_5
# x_29 => mul_22
# x_se_4 => mean_1
# x_se_5 => convolution_7
# x_se_6 => mul_21, sigmoid_6
# x_se_7 => convolution_8
triton_poi_fused_convolution_mean_mul_sigmoid_silu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/ciftnbsatbg3qasxhiplfizxaoy5hzbadox2vrmow7fuzjxug7ka.py
# Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_31 => add_11, mul_24, mul_25, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsi3kqfrphzqoj23dib3zvwmlyw6ccwxuipkat2d2hctbu5z2xk.py
# Source Nodes: [x_36, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_36 => add_13, mul_27, mul_28, sub_6
# x_39 => mul_29, sigmoid_8
triton_poi_fused__native_batch_norm_legit_no_training_silu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cysy2yrq2stvoruzayz4xzd6w2ceezacjqq3zgjtsx2ubyt6rgjc.py
# Source Nodes: [x_41, x_44, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_41 => add_15, mul_31, mul_32, sub_7
# x_44 => mul_33, sigmoid_9
# x_se_8 => mean_2
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 3136
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
        tmp0 = tl.load(in_out_ptr0 + (r2 + (3136*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = 0.001
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
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 3136.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgmcejdoninwzhtzcp7fivadtorle3jn7u3suzetm7zkpgog43p.py
# Source Nodes: [x_44, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_44 => mul_33, sigmoid_9
# x_se_10 => mul_34, sigmoid_10
# x_se_8 => mean_2
# x_se_9 => convolution_12
triton_poi_fused_convolution_mean_silu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_15', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/po/cpopvaewdx7ev4ue2gddahabjfqh4kuzs24if7k2pkoxg6vcskit.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44, x_45, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_44 => mul_33, sigmoid_9
# x_45 => mul_35
# x_se_10 => mul_34, sigmoid_10
# x_se_11 => convolution_13
# x_se_8 => mean_2
# x_se_9 => convolution_12
triton_poi_fused_convolution_mean_mul_sigmoid_silu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckysmawae3xznxf5i7k22xivmbugekehycwrvylowbgfkunn3ib4.py
# Source Nodes: [shortcut_3, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_47 => add_17, mul_37, mul_38, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdxsertwwa22ejprhulrhufbd6umr63vsyrfkaoprud34eu2wc2.py
# Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_53 => add_20, mul_40, mul_41, sub_9
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 144
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyf752z7yqsekwtmwrixyt77bel3ksdh6q5xa23a6hyoq5vegbu.py
# Source Nodes: [x_56, x_58], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_56 => mul_42, sigmoid_12
# x_58 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_silu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 3481
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 59)
    x2 = xindex % 59
    y4 = yindex
    x5 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-57) + x2 + (56*x3) + (3136*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (144*x5) + (501264*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jh/cjhbmdos7p6re4ewxu2trl4w4zj5lxmnyvqgrdhnrog6blbe2ahi.py
# Source Nodes: [x_60, x_63, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_60 => add_22, mul_44, mul_45, sub_10
# x_63 => mul_46, sigmoid_13
# x_se_12 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1152
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
    x0 = xindex % 144
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 784.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33kfn37ahku7eqpy6texczwu5g755y7jr753v64scnx4l7d6okv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63, x_64, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_63 => mul_46, sigmoid_13
# x_64 => mul_48
# x_se_12 => mean_3
# x_se_13 => convolution_17
# x_se_14 => mul_47, sigmoid_14
# x_se_15 => convolution_18
triton_poi_fused_convolution_mean_mul_sigmoid_silu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4q6ebfombig3c47ofk5dbahx4iv3u4mr3odsbvgf72lvkmspm7.py
# Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_66 => add_24, mul_50, mul_51, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bc/cbcze77a6ic7s7yd364gj4qkhpxjpjueozx4hjsa5tsiumrc4she.py
# Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_71 => add_26, mul_53, mul_54, sub_12
# x_74 => mul_55, sigmoid_16
triton_poi_fused__native_batch_norm_legit_no_training_silu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vxvb7mmmprlvroo6cxaqy3rlurmldah7mlj6kdglmcggahuwbk.py
# Source Nodes: [x_76, x_79, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_76 => add_28, mul_57, mul_58, sub_13
# x_79 => mul_59, sigmoid_17
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1920
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
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 784.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckiqsjg6nkp64x652zw7qrdffgox7472tkuwqlkh2pxpfkpkck7z.py
# Source Nodes: [x_79, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_79 => mul_59, sigmoid_17
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
triton_poi_fused_convolution_mean_silu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_25', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmiqcwew2y73taze2an3jw4qx2t4nqrcfr7lscxewh5oqpi4gl6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79, x_80, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_79 => mul_59, sigmoid_17
# x_80 => mul_61
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
# x_se_19 => convolution_23
triton_poi_fused_convolution_mean_mul_sigmoid_silu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/om/comdw3ebm3hx7n6uhmnxjv7cbhezuzevfqruhuuigg6n2pfgnu3p.py
# Source Nodes: [shortcut_5, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_31
# x_82 => add_30, mul_63, mul_64, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 40
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (40*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/an/canihlc4exudxwm3jsossf2zgxgvetgqpmskzktj2yaxaq7cqvrl.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_88 => add_33, mul_66, mul_67, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 240
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfzkf7pahmrubo7utd3is6m2qo46ifootpvtlmoe3giyveo5olo.py
# Source Nodes: [x_91, x_93], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_91 => mul_68, sigmoid_20
# x_93 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_silu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 841
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 29)
    x2 = xindex % 29
    y4 = yindex
    x5 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 28, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x2
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (28*x3) + (784*y4)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (y0 + (240*x5) + (201840*y1)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijdflrro2ltyjaehst2z6xhrkwqulu6ycwm5ag6ype6q6tenrs7.py
# Source Nodes: [x_95, x_98, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_95 => add_35, mul_70, mul_71, sub_16
# x_98 => mul_72, sigmoid_21
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5vvwmvm7pr5zld77s2f4vxgddgfrcviicvhjbsldtchrrncupf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98, x_99, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_98 => mul_72, sigmoid_21
# x_99 => mul_74
# x_se_20 => mean_5
# x_se_21 => convolution_27
# x_se_22 => mul_73, sigmoid_22
# x_se_23 => convolution_28
triton_poi_fused_convolution_mean_mul_sigmoid_silu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfizkcujnpwxfo2nkwk7jc435r77qwicfxf5il7cyvsapl7l4wvv.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_101 => add_37, mul_76, mul_77, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnfazlhzegay4gogb3pgkvskyn7upubev6hwsg4r5q4qvd6sateq.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_106 => add_39, mul_79, mul_80, sub_18
# x_109 => mul_81, sigmoid_24
triton_poi_fused__native_batch_norm_legit_no_training_silu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocvo3rpytyip6k67tljhmrcmt5iqw4ueggogl2gc5qcfgtf7b37.py
# Source Nodes: [x_111, x_114, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_111 => add_41, mul_83, mul_84, sub_19
# x_114 => mul_85, sigmoid_25
# x_se_24 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrwkws2hwgpn7qqhy74b77oe7kphasvgkhj4mtwwv6jdbfx2fuc.py
# Source Nodes: [x_114, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_114 => mul_85, sigmoid_25
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
triton_poi_fused_convolution_mean_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_35', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmfcqi4s5gffdwygep4vtjjw66b3rbuqcpcbibitfb3f6rtaxmj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114, x_115, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_114 => mul_85, sigmoid_25
# x_115 => mul_87
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
# x_se_27 => convolution_33
triton_poi_fused_convolution_mean_mul_sigmoid_silu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehh2sthbljbsbhwixt5rwgs4zfg4urtuftu6kcvrfudgcl2gu5k.py
# Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_7 => add_44
# x_117 => add_43, mul_89, mul_90, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_add_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 80
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpzpue6usbeq3bk77vcojpkqbhum3fp744ucjrm62lcmr45un5o.py
# Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_151 => add_57, mul_115, mul_116, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mr/cmrj5lvyr7xixdfxp3dnctwq6k3m6wg3uvponfkr4m5bmsyejlmd.py
# Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_156 => add_59, mul_118, mul_119, sub_27
# x_159 => mul_120, sigmoid_36
triton_poi_fused__native_batch_norm_legit_no_training_silu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capvfgcrtu72vxlw7mzeizkdymj2io3xsacjlfs7zbvdeho6ztat.py
# Source Nodes: [x_161, x_164, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_161 => add_61, mul_122, mul_123, sub_28
# x_164 => mul_124, sigmoid_37
# x_se_36 => mean_9
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4lg5nt4p3bcl3u3onyyldeu4ovf4pwyvjd5n7v7bwxqbflwp5x.py
# Source Nodes: [x_164, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_164 => mul_124, sigmoid_37
# x_se_36 => mean_9
# x_se_37 => convolution_47
# x_se_38 => mul_125, sigmoid_38
triton_poi_fused_convolution_mean_silu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_41', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/23/c23b6yx66as4h5bwws63fg4tw4pwm5gn6qyy6f24l6tnjedthctc.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164, x_165, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_164 => mul_124, sigmoid_37
# x_165 => mul_126
# x_se_36 => mean_9
# x_se_37 => convolution_47
# x_se_38 => mul_125, sigmoid_38
# x_se_39 => convolution_48
triton_poi_fused_convolution_mean_mul_sigmoid_silu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwexemfjkqbm3rcrae62icif2lsyaeli26krqbmbbakaunzlohz5.py
# Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_10 => add_64
# x_167 => add_63, mul_128, mul_129, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_add_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5s22ij3j4qw4zbmd667lbvtxpzwpsowo6bqi424bkj42pfkh3dl.py
# Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_190 => add_73, mul_144, mul_145, sub_33
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 672
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/op/coppj24xhalfkwnlbc6uxjl67sxu27pv3wq4r2fbdzbik5i3yidx.py
# Source Nodes: [x_193, x_195], Original ATen: [aten.constant_pad_nd, aten.silu]
# x_193 => mul_146, sigmoid_44
# x_195 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_silu_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 17)
    x2 = xindex % 17
    y4 = yindex
    x5 = xindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-15) + x2 + (14*x3) + (196*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (672*x5) + (194208*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csaglecdopvah6foxz6hvfrh6tw7clpauwweo6uvt52hjhodpz3y.py
# Source Nodes: [x_197, x_200, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_197 => add_75, mul_148, mul_149, sub_34
# x_200 => mul_150, sigmoid_45
# x_se_44 => mean_11
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_46', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5hvjbuavrufn4u3x54hyeth3mes4nzzogysr5eilfaq4trl6mz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200, x_201, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
# x_200 => mul_150, sigmoid_45
# x_201 => mul_152
# x_se_44 => mean_11
# x_se_45 => convolution_57
# x_se_46 => mul_151, sigmoid_46
# x_se_47 => convolution_58
triton_poi_fused_convolution_mean_mul_sigmoid_silu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5huon4zlltlzrnm62xnplair5krbh3wp4arqch55j4kbwq3ues.py
# Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_203 => add_77, mul_154, mul_155, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7duotke56g36d2wwd5di5bchcreut7foohrxho5nuhcy3pc2ibl.py
# Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_208 => add_79, mul_157, mul_158, sub_36
# x_211 => mul_159, sigmoid_48
triton_poi_fused__native_batch_norm_legit_no_training_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 49
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2p43ccfrybawtpfmazoxpzdgpvsc7efsq5qi7jck3y2h3g7jq3i.py
# Source Nodes: [x_213, x_216, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_213 => add_81, mul_161, mul_162, sub_37
# x_216 => mul_163, sigmoid_49
# x_se_48 => mean_12
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkurzntwqslzthzahgp53pncfpt4sbduspa3ksk3uioplihsqn6.py
# Source Nodes: [x_216, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_216 => mul_163, sigmoid_49
# x_se_48 => mean_12
# x_se_49 => convolution_62
# x_se_50 => mul_164, sigmoid_50
triton_poi_fused_convolution_mean_silu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_51', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck535xlach3hurze6vvxtclvdjx7poe7qaquq7tfm365dshf2tdh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216, x_217, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_51
# x_216 => mul_163, sigmoid_49
# x_217 => mul_165
# x_se_48 => mean_12
# x_se_49 => convolution_62
# x_se_50 => mul_164, sigmoid_50
# x_se_51 => convolution_63
triton_poi_fused_convolution_mean_mul_sigmoid_silu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllqrgneznen4gdz5z7r5ewpccaqmn6vxbcgrm7aqrwfk2f7ag44.py
# Source Nodes: [shortcut_13, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_13 => add_84
# x_219 => add_83, mul_167, mul_168, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_add_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ur/curmrfq6wqk664wj2wbua24a5lkrfa6nu3rmaxohxa2uv6wrkdni.py
# Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_270 => add_104, mul_206, mul_207, sub_47
triton_poi_fused__native_batch_norm_legit_no_training_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civewbwcmtaotjvahtmzvnugup7bshfypv5xeotbalk3ermu4tek.py
# Source Nodes: [x_276, x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_276 => add_106, mul_209, mul_210, sub_48
# x_280 => mul_211, sigmoid_64
# x_281 => mean_16
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (24, ), (1, ))
    assert_size_stride(arg14_1, (144, ), (1, ))
    assert_size_stride(arg15_1, (144, ), (1, ))
    assert_size_stride(arg16_1, (144, ), (1, ))
    assert_size_stride(arg17_1, (144, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (24, ), (1, ))
    assert_size_stride(arg20_1, (144, ), (1, ))
    assert_size_stride(arg21_1, (144, ), (1, ))
    assert_size_stride(arg22_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg23_1, (144, ), (1, ))
    assert_size_stride(arg24_1, (144, ), (1, ))
    assert_size_stride(arg25_1, (40, ), (1, ))
    assert_size_stride(arg26_1, (40, ), (1, ))
    assert_size_stride(arg27_1, (240, ), (1, ))
    assert_size_stride(arg28_1, (240, ), (1, ))
    assert_size_stride(arg29_1, (240, ), (1, ))
    assert_size_stride(arg30_1, (240, ), (1, ))
    assert_size_stride(arg31_1, (40, ), (1, ))
    assert_size_stride(arg32_1, (40, ), (1, ))
    assert_size_stride(arg33_1, (240, ), (1, ))
    assert_size_stride(arg34_1, (240, ), (1, ))
    assert_size_stride(arg35_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg36_1, (240, ), (1, ))
    assert_size_stride(arg37_1, (240, ), (1, ))
    assert_size_stride(arg38_1, (80, ), (1, ))
    assert_size_stride(arg39_1, (80, ), (1, ))
    assert_size_stride(arg40_1, (480, ), (1, ))
    assert_size_stride(arg41_1, (480, ), (1, ))
    assert_size_stride(arg42_1, (480, ), (1, ))
    assert_size_stride(arg43_1, (480, ), (1, ))
    assert_size_stride(arg44_1, (80, ), (1, ))
    assert_size_stride(arg45_1, (80, ), (1, ))
    assert_size_stride(arg46_1, (480, ), (1, ))
    assert_size_stride(arg47_1, (480, ), (1, ))
    assert_size_stride(arg48_1, (480, ), (1, ))
    assert_size_stride(arg49_1, (480, ), (1, ))
    assert_size_stride(arg50_1, (80, ), (1, ))
    assert_size_stride(arg51_1, (80, ), (1, ))
    assert_size_stride(arg52_1, (480, ), (1, ))
    assert_size_stride(arg53_1, (480, ), (1, ))
    assert_size_stride(arg54_1, (480, ), (1, ))
    assert_size_stride(arg55_1, (480, ), (1, ))
    assert_size_stride(arg56_1, (112, ), (1, ))
    assert_size_stride(arg57_1, (112, ), (1, ))
    assert_size_stride(arg58_1, (672, ), (1, ))
    assert_size_stride(arg59_1, (672, ), (1, ))
    assert_size_stride(arg60_1, (672, ), (1, ))
    assert_size_stride(arg61_1, (672, ), (1, ))
    assert_size_stride(arg62_1, (112, ), (1, ))
    assert_size_stride(arg63_1, (112, ), (1, ))
    assert_size_stride(arg64_1, (672, ), (1, ))
    assert_size_stride(arg65_1, (672, ), (1, ))
    assert_size_stride(arg66_1, (672, ), (1, ))
    assert_size_stride(arg67_1, (672, ), (1, ))
    assert_size_stride(arg68_1, (112, ), (1, ))
    assert_size_stride(arg69_1, (112, ), (1, ))
    assert_size_stride(arg70_1, (672, ), (1, ))
    assert_size_stride(arg71_1, (672, ), (1, ))
    assert_size_stride(arg72_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg73_1, (672, ), (1, ))
    assert_size_stride(arg74_1, (672, ), (1, ))
    assert_size_stride(arg75_1, (192, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (1152, ), (1, ))
    assert_size_stride(arg78_1, (1152, ), (1, ))
    assert_size_stride(arg79_1, (1152, ), (1, ))
    assert_size_stride(arg80_1, (1152, ), (1, ))
    assert_size_stride(arg81_1, (192, ), (1, ))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (1152, ), (1, ))
    assert_size_stride(arg84_1, (1152, ), (1, ))
    assert_size_stride(arg85_1, (1152, ), (1, ))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (192, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (1152, ), (1, ))
    assert_size_stride(arg90_1, (1152, ), (1, ))
    assert_size_stride(arg91_1, (1152, ), (1, ))
    assert_size_stride(arg92_1, (1152, ), (1, ))
    assert_size_stride(arg93_1, (192, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (1152, ), (1, ))
    assert_size_stride(arg96_1, (1152, ), (1, ))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (1152, ), (1, ))
    assert_size_stride(arg99_1, (320, ), (1, ))
    assert_size_stride(arg100_1, (320, ), (1, ))
    assert_size_stride(arg101_1, (1280, ), (1, ))
    assert_size_stride(arg102_1, (1280, ), (1, ))
    assert_size_stride(arg103_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg104_1, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg105_1, (8, ), (1, ))
    assert_size_stride(arg106_1, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg107_1, (32, ), (1, ))
    assert_size_stride(arg108_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg109_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg110_1, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg111_1, (4, ), (1, ))
    assert_size_stride(arg112_1, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg113_1, (96, ), (1, ))
    assert_size_stride(arg114_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg115_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg116_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg117_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg118_1, (6, ), (1, ))
    assert_size_stride(arg119_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg120_1, (144, ), (1, ))
    assert_size_stride(arg121_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg122_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg123_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg124_1, (6, ), (1, ))
    assert_size_stride(arg125_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg126_1, (144, ), (1, ))
    assert_size_stride(arg127_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg128_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg129_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg130_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg131_1, (10, ), (1, ))
    assert_size_stride(arg132_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg133_1, (240, ), (1, ))
    assert_size_stride(arg134_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg135_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg136_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg137_1, (10, ), (1, ))
    assert_size_stride(arg138_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg139_1, (240, ), (1, ))
    assert_size_stride(arg140_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg141_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg142_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg143_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg144_1, (20, ), (1, ))
    assert_size_stride(arg145_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg146_1, (480, ), (1, ))
    assert_size_stride(arg147_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg148_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg149_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg150_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg151_1, (20, ), (1, ))
    assert_size_stride(arg152_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg153_1, (480, ), (1, ))
    assert_size_stride(arg154_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg155_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg156_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg157_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg158_1, (20, ), (1, ))
    assert_size_stride(arg159_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg160_1, (480, ), (1, ))
    assert_size_stride(arg161_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg162_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg163_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg164_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg165_1, (28, ), (1, ))
    assert_size_stride(arg166_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg167_1, (672, ), (1, ))
    assert_size_stride(arg168_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg169_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg170_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg171_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg172_1, (28, ), (1, ))
    assert_size_stride(arg173_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg174_1, (672, ), (1, ))
    assert_size_stride(arg175_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg176_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg177_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg178_1, (28, ), (1, ))
    assert_size_stride(arg179_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg180_1, (672, ), (1, ))
    assert_size_stride(arg181_1, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg182_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg183_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg184_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg185_1, (48, ), (1, ))
    assert_size_stride(arg186_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg187_1, (1152, ), (1, ))
    assert_size_stride(arg188_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg189_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg190_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg191_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg192_1, (48, ), (1, ))
    assert_size_stride(arg193_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg194_1, (1152, ), (1, ))
    assert_size_stride(arg195_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg196_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg197_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg198_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg199_1, (48, ), (1, ))
    assert_size_stride(arg200_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg201_1, (1152, ), (1, ))
    assert_size_stride(arg202_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg203_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg204_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg205_1, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg206_1, (48, ), (1, ))
    assert_size_stride(arg207_1, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg208_1, (1152, ), (1, ))
    assert_size_stride(arg209_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg210_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg211_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg212_1, (1000, ), (1, ))
    assert_size_stride(arg213_1, (32, ), (1, ))
    assert_size_stride(arg214_1, (32, ), (1, ))
    assert_size_stride(arg215_1, (32, ), (1, ))
    assert_size_stride(arg216_1, (32, ), (1, ))
    assert_size_stride(arg217_1, (16, ), (1, ))
    assert_size_stride(arg218_1, (16, ), (1, ))
    assert_size_stride(arg219_1, (96, ), (1, ))
    assert_size_stride(arg220_1, (96, ), (1, ))
    assert_size_stride(arg221_1, (96, ), (1, ))
    assert_size_stride(arg222_1, (96, ), (1, ))
    assert_size_stride(arg223_1, (24, ), (1, ))
    assert_size_stride(arg224_1, (24, ), (1, ))
    assert_size_stride(arg225_1, (144, ), (1, ))
    assert_size_stride(arg226_1, (144, ), (1, ))
    assert_size_stride(arg227_1, (144, ), (1, ))
    assert_size_stride(arg228_1, (144, ), (1, ))
    assert_size_stride(arg229_1, (24, ), (1, ))
    assert_size_stride(arg230_1, (24, ), (1, ))
    assert_size_stride(arg231_1, (144, ), (1, ))
    assert_size_stride(arg232_1, (144, ), (1, ))
    assert_size_stride(arg233_1, (144, ), (1, ))
    assert_size_stride(arg234_1, (144, ), (1, ))
    assert_size_stride(arg235_1, (40, ), (1, ))
    assert_size_stride(arg236_1, (40, ), (1, ))
    assert_size_stride(arg237_1, (240, ), (1, ))
    assert_size_stride(arg238_1, (240, ), (1, ))
    assert_size_stride(arg239_1, (240, ), (1, ))
    assert_size_stride(arg240_1, (240, ), (1, ))
    assert_size_stride(arg241_1, (40, ), (1, ))
    assert_size_stride(arg242_1, (40, ), (1, ))
    assert_size_stride(arg243_1, (240, ), (1, ))
    assert_size_stride(arg244_1, (240, ), (1, ))
    assert_size_stride(arg245_1, (240, ), (1, ))
    assert_size_stride(arg246_1, (240, ), (1, ))
    assert_size_stride(arg247_1, (80, ), (1, ))
    assert_size_stride(arg248_1, (80, ), (1, ))
    assert_size_stride(arg249_1, (480, ), (1, ))
    assert_size_stride(arg250_1, (480, ), (1, ))
    assert_size_stride(arg251_1, (480, ), (1, ))
    assert_size_stride(arg252_1, (480, ), (1, ))
    assert_size_stride(arg253_1, (80, ), (1, ))
    assert_size_stride(arg254_1, (80, ), (1, ))
    assert_size_stride(arg255_1, (480, ), (1, ))
    assert_size_stride(arg256_1, (480, ), (1, ))
    assert_size_stride(arg257_1, (480, ), (1, ))
    assert_size_stride(arg258_1, (480, ), (1, ))
    assert_size_stride(arg259_1, (80, ), (1, ))
    assert_size_stride(arg260_1, (80, ), (1, ))
    assert_size_stride(arg261_1, (480, ), (1, ))
    assert_size_stride(arg262_1, (480, ), (1, ))
    assert_size_stride(arg263_1, (480, ), (1, ))
    assert_size_stride(arg264_1, (480, ), (1, ))
    assert_size_stride(arg265_1, (112, ), (1, ))
    assert_size_stride(arg266_1, (112, ), (1, ))
    assert_size_stride(arg267_1, (672, ), (1, ))
    assert_size_stride(arg268_1, (672, ), (1, ))
    assert_size_stride(arg269_1, (672, ), (1, ))
    assert_size_stride(arg270_1, (672, ), (1, ))
    assert_size_stride(arg271_1, (112, ), (1, ))
    assert_size_stride(arg272_1, (112, ), (1, ))
    assert_size_stride(arg273_1, (672, ), (1, ))
    assert_size_stride(arg274_1, (672, ), (1, ))
    assert_size_stride(arg275_1, (672, ), (1, ))
    assert_size_stride(arg276_1, (672, ), (1, ))
    assert_size_stride(arg277_1, (112, ), (1, ))
    assert_size_stride(arg278_1, (112, ), (1, ))
    assert_size_stride(arg279_1, (672, ), (1, ))
    assert_size_stride(arg280_1, (672, ), (1, ))
    assert_size_stride(arg281_1, (672, ), (1, ))
    assert_size_stride(arg282_1, (672, ), (1, ))
    assert_size_stride(arg283_1, (192, ), (1, ))
    assert_size_stride(arg284_1, (192, ), (1, ))
    assert_size_stride(arg285_1, (1152, ), (1, ))
    assert_size_stride(arg286_1, (1152, ), (1, ))
    assert_size_stride(arg287_1, (1152, ), (1, ))
    assert_size_stride(arg288_1, (1152, ), (1, ))
    assert_size_stride(arg289_1, (192, ), (1, ))
    assert_size_stride(arg290_1, (192, ), (1, ))
    assert_size_stride(arg291_1, (1152, ), (1, ))
    assert_size_stride(arg292_1, (1152, ), (1, ))
    assert_size_stride(arg293_1, (1152, ), (1, ))
    assert_size_stride(arg294_1, (1152, ), (1, ))
    assert_size_stride(arg295_1, (192, ), (1, ))
    assert_size_stride(arg296_1, (192, ), (1, ))
    assert_size_stride(arg297_1, (1152, ), (1, ))
    assert_size_stride(arg298_1, (1152, ), (1, ))
    assert_size_stride(arg299_1, (1152, ), (1, ))
    assert_size_stride(arg300_1, (1152, ), (1, ))
    assert_size_stride(arg301_1, (192, ), (1, ))
    assert_size_stride(arg302_1, (192, ), (1, ))
    assert_size_stride(arg303_1, (1152, ), (1, ))
    assert_size_stride(arg304_1, (1152, ), (1, ))
    assert_size_stride(arg305_1, (1152, ), (1, ))
    assert_size_stride(arg306_1, (1152, ), (1, ))
    assert_size_stride(arg307_1, (320, ), (1, ))
    assert_size_stride(arg308_1, (320, ), (1, ))
    assert_size_stride(arg309_1, (1280, ), (1, ))
    assert_size_stride(arg310_1, (1280, ), (1, ))
    assert_size_stride(arg311_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(arg311_1, buf0, 24, 50625, grid=grid(24, 50625), stream=stream0)
        del arg311_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg213_1, arg214_1, arg1_1, arg2_1, buf4, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg1_1
        del arg213_1
        del arg214_1
        del arg2_1
        del buf3
        # Source Nodes: [shortcut, x_6], Original ATen: [aten.convolution, aten.silu]
        buf5 = extern_kernels.convolution(buf4, arg103_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg103_1
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided((8, 32, 1, 1), (32, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf8 = reinterpret_tensor(buf7, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf7  # reuse
        # Source Nodes: [x_10, x_7, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3.run(buf6, buf8, arg215_1, arg216_1, arg3_1, arg4_1, 256, 12544, grid=grid(256), stream=stream0)
        del arg215_1
        del arg216_1
        del arg3_1
        del arg4_1
        # Source Nodes: [x_10, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf9 = extern_kernels.convolution(buf8, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 8, 1, 1), (8, 1, 1, 1))
        del arg104_1
        del buf8
        buf10 = reinterpret_tensor(buf9, (8, 8, 1, 1), (8, 1, 8, 8), 0); del buf9  # reuse
        # Source Nodes: [x_10, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_4.run(buf10, arg105_1, 64, grid=grid(64), stream=stream0)
        del arg105_1
        # Source Nodes: [x_10, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf11 = extern_kernels.convolution(buf10, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg106_1
        del buf10
        buf12 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_5.run(buf6, buf11, arg107_1, buf12, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg107_1
        del buf11
        del buf6
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_12, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf13 = extern_kernels.convolution(buf12, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg108_1
        del buf12
        buf14 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf13, arg217_1, arg218_1, arg5_1, arg6_1, buf14, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg217_1
        del arg218_1
        del arg5_1
        del arg6_1
        del buf13
        # Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg109_1
        del buf14
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf16, arg219_1, arg220_1, arg7_1, arg8_1, 9633792, grid=grid(9633792), stream=stream0)
        del arg219_1
        del arg220_1
        del arg7_1
        del arg8_1
        buf17 = empty_strided((8, 96, 113, 113), (1225824, 1, 10848, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_23], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_8.run(buf16, buf17, 768, 12769, grid=grid(768, 12769), stream=stream0)
        del buf16
        # Source Nodes: [x_21, x_23, x_24], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.silu]
        buf18 = extern_kernels.convolution(buf17, arg9_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf18, (8, 96, 56, 56), (301056, 3136, 56, 1))
        del arg9_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        buf20 = empty_strided((8, 96, 1, 1), (96, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf20, (8, 96, 1, 1), (96, 1, 96, 96), 0); del buf20  # reuse
        # Source Nodes: [x_25, x_28, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_9.run(buf19, buf21, arg221_1, arg222_1, arg10_1, arg11_1, 768, 3136, grid=grid(768), stream=stream0)
        del arg10_1
        del arg11_1
        del arg221_1
        del arg222_1
        # Source Nodes: [x_28, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf22 = extern_kernels.convolution(buf21, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 4, 1, 1), (4, 1, 1, 1))
        del arg110_1
        del buf21
        buf23 = reinterpret_tensor(buf22, (8, 4, 1, 1), (4, 1, 4, 4), 0); del buf22  # reuse
        # Source Nodes: [x_28, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_10.run(buf23, arg111_1, 32, grid=grid(32), stream=stream0)
        del arg111_1
        # Source Nodes: [x_28, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf24 = extern_kernels.convolution(buf23, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 96, 1, 1), (96, 1, 1, 1))
        del arg112_1
        del buf23
        buf25 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28, x_29, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_11.run(buf19, buf24, arg113_1, buf25, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del arg113_1
        del buf19
        del buf24
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_28, x_29, x_30, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf26 = extern_kernels.convolution(buf25, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg114_1
        del buf25
        buf27 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf26, arg223_1, arg224_1, arg12_1, arg13_1, buf27, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg223_1
        del arg224_1
        del buf26
        # Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg115_1
        buf29 = buf28; del buf28  # reuse
        buf30 = empty_strided((8, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf29, arg225_1, arg226_1, arg14_1, arg15_1, buf30, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg225_1
        del arg226_1
        del buf29
        # Source Nodes: [x_39, x_40], Original ATen: [aten.convolution, aten.silu]
        buf31 = extern_kernels.convolution(buf30, arg116_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf31, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg116_1
        buf32 = buf31; del buf31  # reuse
        buf33 = empty_strided((8, 144, 1, 1), (144, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf34 = reinterpret_tensor(buf33, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf33  # reuse
        # Source Nodes: [x_41, x_44, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_14.run(buf32, buf34, arg227_1, arg228_1, arg16_1, arg17_1, 1152, 3136, grid=grid(1152), stream=stream0)
        del arg16_1
        del arg17_1
        del arg227_1
        del arg228_1
        # Source Nodes: [x_44, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf35 = extern_kernels.convolution(buf34, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg117_1
        del buf34
        buf36 = reinterpret_tensor(buf35, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf35  # reuse
        # Source Nodes: [x_44, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_15.run(buf36, arg118_1, 48, grid=grid(48), stream=stream0)
        del arg118_1
        # Source Nodes: [x_44, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf37 = extern_kernels.convolution(buf36, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg119_1
        del buf36
        buf38 = buf30; del buf30  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44, x_45, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_16.run(buf32, buf37, arg120_1, buf38, 1152, 3136, grid=grid(1152, 3136), stream=stream0)
        del arg120_1
        del buf32
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_44, x_45, x_46, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf39 = extern_kernels.convolution(buf38, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg121_1
        del buf38
        buf40 = buf27; del buf27  # reuse
        # Source Nodes: [shortcut_3, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf40, buf39, arg229_1, arg230_1, arg18_1, arg19_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg18_1
        del arg19_1
        del arg229_1
        del arg230_1
        del buf39
        # Source Nodes: [shortcut_3, x_47, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 144, 56, 56), (451584, 3136, 56, 1))
        del arg122_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf42, arg231_1, arg232_1, arg20_1, arg21_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg20_1
        del arg21_1
        del arg231_1
        del arg232_1
        buf43 = empty_strided((8, 144, 59, 59), (501264, 1, 8496, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56, x_58], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_19.run(buf42, buf43, 1152, 3481, grid=grid(1152, 3481), stream=stream0)
        del buf42
        # Source Nodes: [x_56, x_58, x_59], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.silu]
        buf44 = extern_kernels.convolution(buf43, arg22_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf44, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg22_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        buf46 = reinterpret_tensor(buf37, (8, 144, 1, 1), (144, 1, 1152, 1152), 0); del buf37  # reuse
        buf47 = reinterpret_tensor(buf46, (8, 144, 1, 1), (144, 1, 144, 144), 0); del buf46  # reuse
        # Source Nodes: [x_60, x_63, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_20.run(buf45, buf47, arg233_1, arg234_1, arg23_1, arg24_1, 1152, 784, grid=grid(1152), stream=stream0)
        del arg233_1
        del arg234_1
        del arg23_1
        del arg24_1
        # Source Nodes: [x_63, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf48 = extern_kernels.convolution(buf47, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 6, 1, 1), (6, 1, 1, 1))
        del arg123_1
        del buf47
        buf49 = reinterpret_tensor(buf48, (8, 6, 1, 1), (6, 1, 6, 6), 0); del buf48  # reuse
        # Source Nodes: [x_63, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_15.run(buf49, arg124_1, 48, grid=grid(48), stream=stream0)
        del arg124_1
        # Source Nodes: [x_63, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf50 = extern_kernels.convolution(buf49, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 144, 1, 1), (144, 1, 1, 1))
        del arg125_1
        del buf49
        buf51 = empty_strided((8, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63, x_64, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_21.run(buf45, buf50, arg126_1, buf51, 1152, 784, grid=grid(1152, 784), stream=stream0)
        del arg126_1
        del buf45
        del buf50
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_63, x_64, x_65, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf52 = extern_kernels.convolution(buf51, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg127_1
        del buf51
        buf53 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf52, arg235_1, arg236_1, arg25_1, arg26_1, buf53, 320, 784, grid=grid(320, 784), stream=stream0)
        del arg235_1
        del arg236_1
        del arg25_1
        del arg26_1
        del buf52
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 240, 28, 28), (188160, 784, 28, 1))
        del arg128_1
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_23.run(buf55, arg237_1, arg238_1, arg27_1, arg28_1, buf56, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg237_1
        del arg238_1
        del arg27_1
        del arg28_1
        del buf55
        # Source Nodes: [x_74, x_75], Original ATen: [aten.convolution, aten.silu]
        buf57 = extern_kernels.convolution(buf56, arg129_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf57, (8, 240, 28, 28), (188160, 784, 28, 1))
        del arg129_1
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf60 = reinterpret_tensor(buf59, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf59  # reuse
        # Source Nodes: [x_76, x_79, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_24.run(buf58, buf60, arg239_1, arg240_1, arg29_1, arg30_1, 1920, 784, grid=grid(1920), stream=stream0)
        del arg239_1
        del arg240_1
        del arg29_1
        del arg30_1
        # Source Nodes: [x_79, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf61 = extern_kernels.convolution(buf60, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg130_1
        del buf60
        buf62 = reinterpret_tensor(buf61, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf61  # reuse
        # Source Nodes: [x_79, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_25.run(buf62, arg131_1, 80, grid=grid(80), stream=stream0)
        del arg131_1
        # Source Nodes: [x_79, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf63 = extern_kernels.convolution(buf62, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg132_1
        del buf62
        buf64 = buf56; del buf56  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79, x_80, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_26.run(buf58, buf63, arg133_1, buf64, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg133_1
        del buf58
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_79, x_80, x_81, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf65 = extern_kernels.convolution(buf64, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg134_1
        del buf64
        buf66 = buf53; del buf53  # reuse
        # Source Nodes: [shortcut_5, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_27.run(buf66, buf65, arg241_1, arg242_1, arg31_1, arg32_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg241_1
        del arg242_1
        del arg31_1
        del arg32_1
        del buf65
        # Source Nodes: [shortcut_5, x_82, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 240, 28, 28), (188160, 784, 28, 1))
        del arg135_1
        del buf66
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf68, arg243_1, arg244_1, arg33_1, arg34_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg243_1
        del arg244_1
        del arg33_1
        del arg34_1
        buf69 = empty_strided((8, 240, 29, 29), (201840, 1, 6960, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91, x_93], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_29.run(buf68, buf69, 1920, 841, grid=grid(1920, 841), stream=stream0)
        del buf68
        # Source Nodes: [x_91, x_93, x_94], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.silu]
        buf70 = extern_kernels.convolution(buf69, arg35_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf70, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg35_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        buf72 = reinterpret_tensor(buf63, (8, 240, 1, 1), (240, 1, 1920, 1920), 0); del buf63  # reuse
        buf73 = reinterpret_tensor(buf72, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf72  # reuse
        # Source Nodes: [x_95, x_98, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30.run(buf71, buf73, arg245_1, arg246_1, arg36_1, arg37_1, 1920, 196, grid=grid(1920), stream=stream0)
        del arg245_1
        del arg246_1
        del arg36_1
        del arg37_1
        # Source Nodes: [x_98, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf74 = extern_kernels.convolution(buf73, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 10, 1, 1), (10, 1, 1, 1))
        del arg136_1
        del buf73
        buf75 = reinterpret_tensor(buf74, (8, 10, 1, 1), (10, 1, 10, 10), 0); del buf74  # reuse
        # Source Nodes: [x_98, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_25.run(buf75, arg137_1, 80, grid=grid(80), stream=stream0)
        del arg137_1
        # Source Nodes: [x_98, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf76 = extern_kernels.convolution(buf75, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg138_1
        del buf75
        buf77 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_98, x_99, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_31.run(buf71, buf76, arg139_1, buf77, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg139_1
        del buf71
        del buf76
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_100, x_98, x_99, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf78 = extern_kernels.convolution(buf77, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg140_1
        del buf77
        buf79 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf78, arg247_1, arg248_1, arg38_1, arg39_1, buf79, 640, 196, grid=grid(640, 196), stream=stream0)
        del arg247_1
        del arg248_1
        del arg38_1
        del arg39_1
        del buf78
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg141_1
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_33.run(buf81, arg249_1, arg250_1, arg40_1, arg41_1, buf82, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg249_1
        del arg250_1
        del arg40_1
        del arg41_1
        del buf81
        # Source Nodes: [x_109, x_110], Original ATen: [aten.convolution, aten.silu]
        buf83 = extern_kernels.convolution(buf82, arg142_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf83, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg142_1
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf86 = reinterpret_tensor(buf85, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf85  # reuse
        # Source Nodes: [x_111, x_114, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_34.run(buf84, buf86, arg251_1, arg252_1, arg42_1, arg43_1, 3840, 196, grid=grid(3840), stream=stream0)
        del arg251_1
        del arg252_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_114, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf87 = extern_kernels.convolution(buf86, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg143_1
        del buf86
        buf88 = reinterpret_tensor(buf87, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf87  # reuse
        # Source Nodes: [x_114, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_35.run(buf88, arg144_1, 160, grid=grid(160), stream=stream0)
        del arg144_1
        # Source Nodes: [x_114, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf89 = extern_kernels.convolution(buf88, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg145_1
        del buf88
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114, x_115, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_36.run(buf84, buf89, arg146_1, buf90, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg146_1
        del buf84
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_114, x_115, x_116, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf91 = extern_kernels.convolution(buf90, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg147_1
        buf92 = buf79; del buf79  # reuse
        # Source Nodes: [shortcut_7, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf92, buf91, arg253_1, arg254_1, arg44_1, arg45_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg253_1
        del arg254_1
        del arg44_1
        del arg45_1
        del buf91
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg148_1
        buf94 = buf93; del buf93  # reuse
        buf95 = buf90; del buf90  # reuse
        # Source Nodes: [x_123, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_33.run(buf94, arg255_1, arg256_1, arg46_1, arg47_1, buf95, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg255_1
        del arg256_1
        del arg46_1
        del arg47_1
        del buf94
        # Source Nodes: [x_126, x_127], Original ATen: [aten.convolution, aten.silu]
        buf96 = extern_kernels.convolution(buf95, arg149_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf96, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg149_1
        buf97 = buf96; del buf96  # reuse
        buf98 = reinterpret_tensor(buf89, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf89  # reuse
        buf99 = reinterpret_tensor(buf98, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf98  # reuse
        # Source Nodes: [x_128, x_131, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_34.run(buf97, buf99, arg257_1, arg258_1, arg48_1, arg49_1, 3840, 196, grid=grid(3840), stream=stream0)
        del arg257_1
        del arg258_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_131, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf100 = extern_kernels.convolution(buf99, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg150_1
        del buf99
        buf101 = reinterpret_tensor(buf100, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf100  # reuse
        # Source Nodes: [x_131, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_35.run(buf101, arg151_1, 160, grid=grid(160), stream=stream0)
        del arg151_1
        # Source Nodes: [x_131, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf102 = extern_kernels.convolution(buf101, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg152_1
        del buf101
        buf103 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_131, x_132, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_36.run(buf97, buf102, arg153_1, buf103, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg153_1
        del buf97
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_131, x_132, x_133, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf104 = extern_kernels.convolution(buf103, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg154_1
        buf105 = buf92; del buf92  # reuse
        # Source Nodes: [shortcut_8, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf105, buf104, arg259_1, arg260_1, arg50_1, arg51_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg259_1
        del arg260_1
        del arg50_1
        del arg51_1
        del buf104
        # Source Nodes: [shortcut_8, x_134, x_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg155_1
        buf107 = buf106; del buf106  # reuse
        buf108 = buf103; del buf103  # reuse
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_33.run(buf107, arg261_1, arg262_1, arg52_1, arg53_1, buf108, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg261_1
        del arg262_1
        del arg52_1
        del arg53_1
        del buf107
        # Source Nodes: [x_143, x_144], Original ATen: [aten.convolution, aten.silu]
        buf109 = extern_kernels.convolution(buf108, arg156_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf109, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg156_1
        buf110 = buf109; del buf109  # reuse
        buf111 = reinterpret_tensor(buf102, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf102  # reuse
        buf112 = reinterpret_tensor(buf111, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf111  # reuse
        # Source Nodes: [x_145, x_148, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_34.run(buf110, buf112, arg263_1, arg264_1, arg54_1, arg55_1, 3840, 196, grid=grid(3840), stream=stream0)
        del arg263_1
        del arg264_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_148, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf113 = extern_kernels.convolution(buf112, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 20, 1, 1), (20, 1, 1, 1))
        del arg157_1
        del buf112
        buf114 = reinterpret_tensor(buf113, (8, 20, 1, 1), (20, 1, 20, 20), 0); del buf113  # reuse
        # Source Nodes: [x_148, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_35.run(buf114, arg158_1, 160, grid=grid(160), stream=stream0)
        del arg158_1
        # Source Nodes: [x_148, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf115 = extern_kernels.convolution(buf114, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg159_1
        del buf114
        buf116 = buf108; del buf108  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_148, x_149, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_36.run(buf110, buf115, arg160_1, buf116, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg160_1
        del buf110
        del buf115
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_148, x_149, x_150, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf117 = extern_kernels.convolution(buf116, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg161_1
        del buf116
        buf118 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf117, arg265_1, arg266_1, arg56_1, arg57_1, buf118, 896, 196, grid=grid(896, 196), stream=stream0)
        del arg265_1
        del arg266_1
        del arg56_1
        del arg57_1
        del buf117
        # Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg162_1
        buf120 = buf119; del buf119  # reuse
        buf121 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_39.run(buf120, arg267_1, arg268_1, arg58_1, arg59_1, buf121, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg267_1
        del arg268_1
        del arg58_1
        del arg59_1
        del buf120
        # Source Nodes: [x_159, x_160], Original ATen: [aten.convolution, aten.silu]
        buf122 = extern_kernels.convolution(buf121, arg163_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf122, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg163_1
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf125 = reinterpret_tensor(buf124, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf124  # reuse
        # Source Nodes: [x_161, x_164, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40.run(buf123, buf125, arg269_1, arg270_1, arg60_1, arg61_1, 5376, 196, grid=grid(5376), stream=stream0)
        del arg269_1
        del arg270_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_164, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf126 = extern_kernels.convolution(buf125, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg164_1
        del buf125
        buf127 = reinterpret_tensor(buf126, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf126  # reuse
        # Source Nodes: [x_164, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_41.run(buf127, arg165_1, 224, grid=grid(224), stream=stream0)
        del arg165_1
        # Source Nodes: [x_164, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf128 = extern_kernels.convolution(buf127, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg166_1
        del buf127
        buf129 = buf121; del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164, x_165, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_42.run(buf123, buf128, arg167_1, buf129, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg167_1
        del buf123
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_164, x_165, x_166, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf130 = extern_kernels.convolution(buf129, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg168_1
        buf131 = buf118; del buf118  # reuse
        # Source Nodes: [shortcut_10, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_43.run(buf131, buf130, arg271_1, arg272_1, arg62_1, arg63_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg271_1
        del arg272_1
        del arg62_1
        del arg63_1
        del buf130
        # Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg169_1
        buf133 = buf132; del buf132  # reuse
        buf134 = buf129; del buf129  # reuse
        # Source Nodes: [x_173, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_39.run(buf133, arg273_1, arg274_1, arg64_1, arg65_1, buf134, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg273_1
        del arg274_1
        del arg64_1
        del arg65_1
        del buf133
        # Source Nodes: [x_176, x_177], Original ATen: [aten.convolution, aten.silu]
        buf135 = extern_kernels.convolution(buf134, arg170_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf135, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg170_1
        buf136 = buf135; del buf135  # reuse
        buf137 = reinterpret_tensor(buf128, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf128  # reuse
        buf138 = reinterpret_tensor(buf137, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf137  # reuse
        # Source Nodes: [x_178, x_181, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40.run(buf136, buf138, arg275_1, arg276_1, arg66_1, arg67_1, 5376, 196, grid=grid(5376), stream=stream0)
        del arg275_1
        del arg276_1
        del arg66_1
        del arg67_1
        # Source Nodes: [x_181, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf139 = extern_kernels.convolution(buf138, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg171_1
        del buf138
        buf140 = reinterpret_tensor(buf139, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf139  # reuse
        # Source Nodes: [x_181, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_41.run(buf140, arg172_1, 224, grid=grid(224), stream=stream0)
        del arg172_1
        # Source Nodes: [x_181, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf141 = extern_kernels.convolution(buf140, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg173_1
        del buf140
        buf142 = buf134; del buf134  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_181, x_182, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_42.run(buf136, buf141, arg174_1, buf142, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg174_1
        del buf136
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_181, x_182, x_183, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf143 = extern_kernels.convolution(buf142, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg175_1
        del buf142
        buf144 = buf131; del buf131  # reuse
        # Source Nodes: [shortcut_11, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_43.run(buf144, buf143, arg277_1, arg278_1, arg68_1, arg69_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg277_1
        del arg278_1
        del arg68_1
        del arg69_1
        del buf143
        # Source Nodes: [shortcut_11, x_184, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg176_1
        del buf144
        buf146 = buf145; del buf145  # reuse
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf146, arg279_1, arg280_1, arg70_1, arg71_1, 1053696, grid=grid(1053696), stream=stream0)
        del arg279_1
        del arg280_1
        del arg70_1
        del arg71_1
        buf147 = empty_strided((8, 672, 17, 17), (194208, 1, 11424, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193, x_195], Original ATen: [aten.constant_pad_nd, aten.silu]
        triton_poi_fused_constant_pad_nd_silu_45.run(buf146, buf147, 5376, 289, grid=grid(5376, 289), stream=stream0)
        del buf146
        # Source Nodes: [x_193, x_195, x_196], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.silu]
        buf148 = extern_kernels.convolution(buf147, arg72_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf148, (8, 672, 7, 7), (32928, 49, 7, 1))
        del arg72_1
        del buf147
        buf149 = buf148; del buf148  # reuse
        buf150 = reinterpret_tensor(buf141, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf141  # reuse
        buf151 = reinterpret_tensor(buf150, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf150  # reuse
        # Source Nodes: [x_197, x_200, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_46.run(buf149, buf151, arg281_1, arg282_1, arg73_1, arg74_1, 5376, 49, grid=grid(5376), stream=stream0)
        del arg281_1
        del arg282_1
        del arg73_1
        del arg74_1
        # Source Nodes: [x_200, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf152 = extern_kernels.convolution(buf151, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 28, 1, 1), (28, 1, 1, 1))
        del arg177_1
        del buf151
        buf153 = reinterpret_tensor(buf152, (8, 28, 1, 1), (28, 1, 28, 28), 0); del buf152  # reuse
        # Source Nodes: [x_200, x_se_44, x_se_45, x_se_46], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_41.run(buf153, arg178_1, 224, grid=grid(224), stream=stream0)
        del arg178_1
        # Source Nodes: [x_200, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf154 = extern_kernels.convolution(buf153, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg179_1
        del buf153
        buf155 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200, x_201, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_47.run(buf149, buf154, arg180_1, buf155, 5376, 49, grid=grid(5376, 49), stream=stream0)
        del arg180_1
        del buf149
        del buf154
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_200, x_201, x_202, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf156 = extern_kernels.convolution(buf155, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg181_1
        del buf155
        buf157 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf156, arg283_1, arg284_1, arg75_1, arg76_1, buf157, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del arg283_1
        del arg284_1
        del arg75_1
        del arg76_1
        del buf156
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg182_1
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided((8, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf159, arg285_1, arg286_1, arg77_1, arg78_1, buf160, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg285_1
        del arg286_1
        del arg77_1
        del arg78_1
        del buf159
        # Source Nodes: [x_211, x_212], Original ATen: [aten.convolution, aten.silu]
        buf161 = extern_kernels.convolution(buf160, arg183_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf161, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg183_1
        buf162 = buf161; del buf161  # reuse
        buf163 = empty_strided((8, 1152, 1, 1), (1152, 1, 9216, 9216), device='cuda', dtype=torch.float32)
        buf164 = reinterpret_tensor(buf163, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf163  # reuse
        # Source Nodes: [x_213, x_216, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50.run(buf162, buf164, arg287_1, arg288_1, arg79_1, arg80_1, 9216, 49, grid=grid(9216), stream=stream0)
        del arg287_1
        del arg288_1
        del arg79_1
        del arg80_1
        # Source Nodes: [x_216, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf165 = extern_kernels.convolution(buf164, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg184_1
        del buf164
        buf166 = reinterpret_tensor(buf165, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf165  # reuse
        # Source Nodes: [x_216, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_51.run(buf166, arg185_1, 384, grid=grid(384), stream=stream0)
        del arg185_1
        # Source Nodes: [x_216, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf167 = extern_kernels.convolution(buf166, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg186_1
        del buf166
        buf168 = buf160; del buf160  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216, x_217, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_52.run(buf162, buf167, arg187_1, buf168, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg187_1
        del buf162
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_216, x_217, x_218, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf169 = extern_kernels.convolution(buf168, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg188_1
        buf170 = buf157; del buf157  # reuse
        # Source Nodes: [shortcut_13, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_53.run(buf170, buf169, arg289_1, arg290_1, arg81_1, arg82_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg289_1
        del arg290_1
        del arg81_1
        del arg82_1
        del buf169
        # Source Nodes: [x_224], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg189_1
        buf172 = buf171; del buf171  # reuse
        buf173 = buf168; del buf168  # reuse
        # Source Nodes: [x_225, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf172, arg291_1, arg292_1, arg83_1, arg84_1, buf173, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg291_1
        del arg292_1
        del arg83_1
        del arg84_1
        del buf172
        # Source Nodes: [x_228, x_229], Original ATen: [aten.convolution, aten.silu]
        buf174 = extern_kernels.convolution(buf173, arg190_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf174, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg190_1
        buf175 = buf174; del buf174  # reuse
        buf176 = reinterpret_tensor(buf167, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf167  # reuse
        buf177 = reinterpret_tensor(buf176, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf176  # reuse
        # Source Nodes: [x_230, x_233, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50.run(buf175, buf177, arg293_1, arg294_1, arg85_1, arg86_1, 9216, 49, grid=grid(9216), stream=stream0)
        del arg293_1
        del arg294_1
        del arg85_1
        del arg86_1
        # Source Nodes: [x_233, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf178 = extern_kernels.convolution(buf177, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg191_1
        del buf177
        buf179 = reinterpret_tensor(buf178, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf178  # reuse
        # Source Nodes: [x_233, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_51.run(buf179, arg192_1, 384, grid=grid(384), stream=stream0)
        del arg192_1
        # Source Nodes: [x_233, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf180 = extern_kernels.convolution(buf179, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg193_1
        del buf179
        buf181 = buf173; del buf173  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_233, x_234, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_52.run(buf175, buf180, arg194_1, buf181, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg194_1
        del buf175
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_233, x_234, x_235, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf182 = extern_kernels.convolution(buf181, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg195_1
        buf183 = buf170; del buf170  # reuse
        # Source Nodes: [shortcut_14, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_53.run(buf183, buf182, arg295_1, arg296_1, arg87_1, arg88_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg295_1
        del arg296_1
        del arg87_1
        del arg88_1
        del buf182
        # Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg196_1
        buf185 = buf184; del buf184  # reuse
        buf186 = buf181; del buf181  # reuse
        # Source Nodes: [x_242, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf185, arg297_1, arg298_1, arg89_1, arg90_1, buf186, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg297_1
        del arg298_1
        del arg89_1
        del arg90_1
        del buf185
        # Source Nodes: [x_245, x_246], Original ATen: [aten.convolution, aten.silu]
        buf187 = extern_kernels.convolution(buf186, arg197_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf187, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg197_1
        buf188 = buf187; del buf187  # reuse
        buf189 = reinterpret_tensor(buf180, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf180  # reuse
        buf190 = reinterpret_tensor(buf189, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf189  # reuse
        # Source Nodes: [x_247, x_250, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50.run(buf188, buf190, arg299_1, arg300_1, arg91_1, arg92_1, 9216, 49, grid=grid(9216), stream=stream0)
        del arg299_1
        del arg300_1
        del arg91_1
        del arg92_1
        # Source Nodes: [x_250, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf191 = extern_kernels.convolution(buf190, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg198_1
        del buf190
        buf192 = reinterpret_tensor(buf191, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf191  # reuse
        # Source Nodes: [x_250, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_51.run(buf192, arg199_1, 384, grid=grid(384), stream=stream0)
        del arg199_1
        # Source Nodes: [x_250, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf193 = extern_kernels.convolution(buf192, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg200_1
        del buf192
        buf194 = buf186; del buf186  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_250, x_251, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_52.run(buf188, buf193, arg201_1, buf194, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg201_1
        del buf188
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_250, x_251, x_252, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf195 = extern_kernels.convolution(buf194, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg202_1
        buf196 = buf183; del buf183  # reuse
        # Source Nodes: [shortcut_15, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_53.run(buf196, buf195, arg301_1, arg302_1, arg93_1, arg94_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg301_1
        del arg302_1
        del arg93_1
        del arg94_1
        del buf195
        # Source Nodes: [shortcut_15, x_253, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg203_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        buf199 = buf194; del buf194  # reuse
        # Source Nodes: [x_259, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_49.run(buf198, arg303_1, arg304_1, arg95_1, arg96_1, buf199, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg303_1
        del arg304_1
        del arg95_1
        del arg96_1
        del buf198
        # Source Nodes: [x_262, x_263], Original ATen: [aten.convolution, aten.silu]
        buf200 = extern_kernels.convolution(buf199, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf200, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg204_1
        buf201 = buf200; del buf200  # reuse
        buf202 = reinterpret_tensor(buf193, (8, 1152, 1, 1), (1152, 1, 9216, 9216), 0); del buf193  # reuse
        buf203 = reinterpret_tensor(buf202, (8, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf202  # reuse
        # Source Nodes: [x_264, x_267, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_50.run(buf201, buf203, arg305_1, arg306_1, arg97_1, arg98_1, 9216, 49, grid=grid(9216), stream=stream0)
        del arg305_1
        del arg306_1
        del arg97_1
        del arg98_1
        # Source Nodes: [x_267, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf204 = extern_kernels.convolution(buf203, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 48, 1, 1), (48, 1, 1, 1))
        del arg205_1
        del buf203
        buf205 = reinterpret_tensor(buf204, (8, 48, 1, 1), (48, 1, 48, 48), 0); del buf204  # reuse
        # Source Nodes: [x_267, x_se_60, x_se_61, x_se_62], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_51.run(buf205, arg206_1, 384, grid=grid(384), stream=stream0)
        del arg206_1
        # Source Nodes: [x_267, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf206 = extern_kernels.convolution(buf205, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1152, 1, 1), (1152, 1, 1, 1))
        del arg207_1
        del buf205
        buf207 = buf199; del buf199  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_267, x_268, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_52.run(buf201, buf206, arg208_1, buf207, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg208_1
        del buf201
        del buf206
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_267, x_268, x_269, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf208 = extern_kernels.convolution(buf207, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 320, 7, 7), (15680, 49, 7, 1))
        del arg209_1
        del buf207
        buf209 = reinterpret_tensor(buf105, (8, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf105  # reuse
        # Source Nodes: [x_270], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_54.run(buf208, arg307_1, arg308_1, arg99_1, arg100_1, buf209, 2560, 49, grid=grid(2560, 49), stream=stream0)
        del arg100_1
        del arg307_1
        del arg308_1
        del arg99_1
        del buf208
        # Source Nodes: [x_270, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf210 = extern_kernels.convolution(buf209, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 1280, 7, 7), (62720, 49, 7, 1))
        del arg210_1
        del buf209
        buf211 = buf210; del buf210  # reuse
        buf212 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf213 = reinterpret_tensor(buf212, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf212  # reuse
        # Source Nodes: [x_276, x_280, x_281], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_55.run(buf211, buf213, arg309_1, arg310_1, arg101_1, arg102_1, 10240, 49, grid=grid(10240), stream=stream0)
        del arg101_1
        del arg102_1
        del arg309_1
        del arg310_1
        del buf211
        buf214 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg212_1, reinterpret_tensor(buf213, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg211_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf214)
        del arg211_1
        del arg212_1
        return (buf214, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_efficientnet_b0', benchmark_compiled_module)
