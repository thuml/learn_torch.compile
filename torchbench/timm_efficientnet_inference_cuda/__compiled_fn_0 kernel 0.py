
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


# kernel path: /tmp/torchinductor_youkaichao/7e/c7edajh3b7r7vld34sx5nslzqlorfl5flgti2jgo4emdfdd6vcyy.py
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
    size_hints=[16, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
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


# kernel path: /tmp/torchinductor_youkaichao/d7/cd75km3ijrknjeayqf5h3nz7jm7yba7ptcgskdgratlhwvfmryif.py
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
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcuhi5srxdsbfyp4zyuobjgocnv7qur5b73b3hrzgkrqpwvkszk.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x1 = (xindex // 2) % 32
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r3 + (6272*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(in_out_ptr0 + (r3 + (6272*x4)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/csw7zrbgzenelic6ethwjdvf4zlsoddmw4tw5obn7yi2bfn3zvso.py
# Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
# x_9 => mul_7, sigmoid_1
# x_se => mean
triton_per_fused_mean_silu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_4', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 12544.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfshr3wbermsdj3rebzoa2xon3petwut2mx2a6krlrcbclggj5f.py
# Source Nodes: [x_9, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_9 => mul_7, sigmoid_1
# x_se => mean
# x_se_1 => convolution_2
# x_se_2 => mul_8, sigmoid_2
triton_poi_fused_convolution_mean_silu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7sta4kaj57gfgwbtlzluozobgebikhiczg2nxgbqer5n4my7ri.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_9
# x_9 => mul_7, sigmoid_1
# x_se => mean
# x_se_1 => convolution_2
# x_se_2 => mul_8, sigmoid_2
# x_se_3 => convolution_3
triton_poi_fused_convolution_mean_mul_sigmoid_silu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7zjo3h6qhdfawktztjbiktedlcqu7veovfnpdgevvrdf2yqswi.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_12 => add_5, mul_11, mul_12, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7fizjtyuzwhti2zx5ffobqb2ejkqnuq265o7ii6cmi5oabnlbq.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_17 => add_7, mul_14, mul_15, sub_3
# x_20 => mul_16, sigmoid_4
triton_poi_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 12544
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ot5y2hgz7eq4pwlolrnalc3cdkiartptenkguxjqrf7qvuyg3c.py
# Source Nodes: [x_22, x_25, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_22 => add_9, mul_18, mul_19, sub_4
# x_25 => mul_20, sigmoid_5
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
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 3136.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5u6chdp6ngjf2wdvj2siiaao73kfct2bxzp2b2yc3frrkxtb3r.py
# Source Nodes: [x_25, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_25 => mul_20, sigmoid_5
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
    size_hints=[16], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/rb/crbr2qoqyuxvz6dfxvgv74dvg7n3co3bp256bbso7z342e5ozlqs.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_25 => mul_20, sigmoid_5
# x_26 => mul_22
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
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/uc/cucp3voyy2os4gcojqygse4mwkg2cdid6iqgiurqx7j52a6j66i6.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_28 => add_11, mul_24, mul_25, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmcklxdonr4ohf6o6ot352li7k466pixcyss4qfhq3zcten4vjr.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_33 => add_13, mul_27, mul_28, sub_6
# x_36 => mul_29, sigmoid_8
triton_poi_fused__native_batch_norm_legit_no_training_silu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cnczes2kbb753ggvvfadxccuwt7fvaemefrkuqw35cmmlponnjqv.py
# Source Nodes: [x_38, x_41, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_38 => add_15, mul_31, mul_32, sub_7
# x_41 => mul_33, sigmoid_9
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
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
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
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp20 = 3136.0
    tmp21 = tmp18 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmqsrbx6zqa4jh5jwyjrgatxlj4w3gabcdk2qaqgrtoktvshhmu.py
# Source Nodes: [x_41, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_41 => mul_33, sigmoid_9
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
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/6a/c6ahx24z2r2qnek5r6iwaqdredobjmummjgrr6xizkqsa3swrvse.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_41 => mul_33, sigmoid_9
# x_42 => mul_35
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
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/be/cbehr2jnqflaifnpti24dfptdo5pqkyju7v7eet4rgb5danbp366.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_44 => add_17, mul_37, mul_38, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
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
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mcfr5igmdqbeupp76dalkh2vcoltsuj7qiw2csc2pzylxvyaue.py
# Source Nodes: [x_55, x_58, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_55 => add_22, mul_44, mul_45, sub_10
# x_58 => mul_46, sigmoid_13
# x_se_12 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 576
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
    tmp21 = 784.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crsrafqk3ikyc6vuaw2sj57ao6ubtehaqzi44dwltm725ohb4zui.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_58 => mul_46, sigmoid_13
# x_59 => mul_48
# x_se_12 => mean_3
# x_se_13 => convolution_17
# x_se_14 => mul_47, sigmoid_14
# x_se_15 => convolution_18
triton_poi_fused_convolution_mean_mul_sigmoid_silu_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjjh7ht247rebt5xauumfjw4j45djomobxepojaod2xlrwlxvw6.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_61 => add_24, mul_50, mul_51, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/yc/cycup7ctpsw5rvtayw564ycdhpj5pakoctyaodiezxxw7ce3f3md.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_66 => add_26, mul_53, mul_54, sub_12
# x_69 => mul_55, sigmoid_16
triton_poi_fused__native_batch_norm_legit_no_training_silu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bm3odjvmayqyewvww4gzj5zpxtspb5ncwwhldnksqdu7pr5b7a.py
# Source Nodes: [x_71, x_74, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_71 => add_28, mul_57, mul_58, sub_13
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 960
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
    tmp21 = 784.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedydzmkzjoym5n4dx3s3qbafrghl3i3lstsg6lt65k2g6zes3g7.py
# Source Nodes: [x_74, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
triton_poi_fused_convolution_mean_silu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rh2cianayba6spcdbnlcynmvjb3qk5cbhygqsyws7psl7fn3y5.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_74 => mul_59, sigmoid_17
# x_75 => mul_61
# x_se_16 => mean_4
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
# x_se_19 => convolution_23
triton_poi_fused_convolution_mean_mul_sigmoid_silu_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uh3y4dws5qlanrvpfhsj6qupurkcdj4kppw2f5b5rb2vpflffi.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_31
# x_77 => add_30, mul_63, mul_64, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
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


# kernel path: /tmp/torchinductor_youkaichao/sm/csmex65zrozbhmfuexjd3kmlh4gx5lhylmldvleata2ozvutobnd.py
# Source Nodes: [x_88, x_91, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_88 => add_35, mul_70, mul_71, sub_16
# x_91 => mul_72, sigmoid_21
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/cziaqpd2r7fxasw65ag2rh7xix7wn656gyoa3klrzgnxb6lqhb6y.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_91 => mul_72, sigmoid_21
# x_92 => mul_74
# x_se_20 => mean_5
# x_se_21 => convolution_27
# x_se_22 => mul_73, sigmoid_22
# x_se_23 => convolution_28
triton_poi_fused_convolution_mean_mul_sigmoid_silu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjagv2odc27g3ytd5e54num5v27h7rmczz6ncat2bkol2q5gtvk.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_94 => add_37, mul_76, mul_77, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/uc/cuce2bllkbq2ov6qjr5s4oghp2sclyb4huflo2ztt3w7yeqmsep7.py
# Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_102 => mul_81, sigmoid_24
# x_99 => add_39, mul_79, mul_80, sub_18
triton_poi_fused__native_batch_norm_legit_no_training_silu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xcygxn6pjk6elqcyozv23wem37hkocx6i4lma56j2np35dekpf.py
# Source Nodes: [x_104, x_107, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_104 => add_41, mul_83, mul_84, sub_19
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
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
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2sodgtlkit4z6qf2hvoegbbgl5nlaa7f2afpx2mk2kfgotwqck.py
# Source Nodes: [x_107, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
triton_poi_fused_convolution_mean_silu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3jvsffkbnnk74i4dnmnem6jmynukr44vs4h7xpllweylwkk3oe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_107 => mul_85, sigmoid_25
# x_108 => mul_87
# x_se_24 => mean_6
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
# x_se_27 => convolution_33
triton_poi_fused_convolution_mean_mul_sigmoid_silu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_32', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wmvtoqj3zuogcx7gyjceezcq7ovd35slg2e7a6roqmhreigi7c.py
# Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_7 => add_44
# x_110 => add_43, mul_89, mul_90, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_add_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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


# kernel path: /tmp/torchinductor_youkaichao/hk/chkrljzbutdky6mokwso5eovlvz7de6c6ocrtpwldosyt4jjzh2c.py
# Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_144 => add_57, mul_115, mul_116, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/py/cpyynsjoiw7iog3k5xai4syfghh73ia5msqqlllf5vrzowzm56mz.py
# Source Nodes: [x_149, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_149 => add_59, mul_118, mul_119, sub_27
# x_152 => mul_120, sigmoid_36
triton_poi_fused__native_batch_norm_legit_no_training_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
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
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctx2yboiymnbbwng7f6q35qhsb4m3hio5i6eaj7iuhqumr4t7mv6.py
# Source Nodes: [x_154, x_157, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_154 => add_61, mul_122, mul_123, sub_28
# x_157 => mul_124, sigmoid_37
# x_se_36 => mean_9
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
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
    tmp21 = 196.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cusighgt5xdnchqhjmqedtx5pj6pwrtte3ud3iq5rtefwqy6l5cb.py
# Source Nodes: [x_157, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_157 => mul_124, sigmoid_37
# x_se_36 => mean_9
# x_se_37 => convolution_47
# x_se_38 => mul_125, sigmoid_38
triton_poi_fused_convolution_mean_silu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
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


# kernel path: /tmp/torchinductor_youkaichao/hv/chvfra32upy5pqpdar7ayatmlt7xykbaoqzjzaqqqpwupr6q4oeq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_157, x_158, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_157 => mul_124, sigmoid_37
# x_158 => mul_126
# x_se_36 => mean_9
# x_se_37 => convolution_47
# x_se_38 => mul_125, sigmoid_38
# x_se_39 => convolution_48
triton_poi_fused_convolution_mean_mul_sigmoid_silu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
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


# kernel path: /tmp/torchinductor_youkaichao/lg/clggw4xzwcl2fpbaeixrpmicqi7xr6yf5btuulo6ypvmjn7yjxmo.py
# Source Nodes: [shortcut_10, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_10 => add_64
# x_160 => add_63, mul_128, mul_129, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_add_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
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


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyzo6ebx3372lw6v5yn3n4fj3rfdjqfwyjr3oaep4cwqdfv4h6k.py
# Source Nodes: [x_188, x_191, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_188 => add_75, mul_148, mul_149, sub_34
# x_191 => mul_150, sigmoid_45
# x_se_44 => mean_11
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyiqon2kwkwz6y6xxsuvvboxt3gv2e7lmjcvc726atjbiaapvbel.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_191, x_192, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
# x_191 => mul_150, sigmoid_45
# x_192 => mul_152
# x_se_44 => mean_11
# x_se_45 => convolution_57
# x_se_46 => mul_151, sigmoid_46
# x_se_47 => convolution_58
triton_poi_fused_convolution_mean_mul_sigmoid_silu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
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


# kernel path: /tmp/torchinductor_youkaichao/pu/cpum6jiqfvcc5w5sqbgj4jkdtmhwe7thsnwbj77cr73perxqbzv7.py
# Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_194 => add_77, mul_154, mul_155, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/2u/c2u3ac2hr6i7pqxbud6t4m2bhi2nrco3a6pzjc2wkxcckpfu6pje.py
# Source Nodes: [x_199, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# x_199 => add_79, mul_157, mul_158, sub_36
# x_202 => mul_159, sigmoid_48
triton_poi_fused__native_batch_norm_legit_no_training_silu_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n5/cn5roglparnq5r526lb4olux6ud5pl4r3otwchovzvv2y5nlisqg.py
# Source Nodes: [x_204, x_207, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_204 => add_81, mul_161, mul_162, sub_37
# x_207 => mul_163, sigmoid_49
# x_se_48 => mean_12
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepplo4qeiruylg3mp3y3bjjanknny5x5ckntsf5s6fbi7q5fn7e.py
# Source Nodes: [x_207, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
# x_207 => mul_163, sigmoid_49
# x_se_48 => mean_12
# x_se_49 => convolution_62
# x_se_50 => mul_164, sigmoid_50
triton_poi_fused_convolution_mean_silu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_silu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5bouhn4adpowumdskkzspfwmyl4y7enotmm2lku6vqepjwhacu.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_207, x_208, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_51
# x_207 => mul_163, sigmoid_49
# x_208 => mul_165
# x_se_48 => mean_12
# x_se_49 => convolution_62
# x_se_50 => mul_164, sigmoid_50
# x_se_51 => convolution_63
triton_poi_fused_convolution_mean_mul_sigmoid_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_mul_sigmoid_silu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ya7n76cve24ghso3h55hzqggjmkavlslwagtc6o3hddnyxsv5w.py
# Source Nodes: [shortcut_13, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_13 => add_84
# x_210 => add_83, mul_167, mul_168, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_add_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
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


# kernel path: /tmp/torchinductor_youkaichao/hx/chx4ok55gs36jqfn76fy4q4wxpbqft7gfprumxeqcx3yiaelmubq.py
# Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_261 => add_104, mul_206, mul_207, sub_47
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
    ynumel = 1280
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/25/c25w6sh5zeetzc5fk3oqiosvdn2u2vlasxx2vhtm6hvdxmc6amb5.py
# Source Nodes: [x_267, x_271, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# x_267 => add_106, mul_209, mul_210, sub_48
# x_271 => mul_211, sigmoid_64
# x_272 => mean_16
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_49', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
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
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
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
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp22, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1 = args
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
    assert_size_stride(arg52_1, (112, ), (1, ))
    assert_size_stride(arg53_1, (112, ), (1, ))
    assert_size_stride(arg54_1, (672, ), (1, ))
    assert_size_stride(arg55_1, (672, ), (1, ))
    assert_size_stride(arg56_1, (672, ), (1, ))
    assert_size_stride(arg57_1, (672, ), (1, ))
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
    assert_size_stride(arg70_1, (192, ), (1, ))
    assert_size_stride(arg71_1, (192, ), (1, ))
    assert_size_stride(arg72_1, (1152, ), (1, ))
    assert_size_stride(arg73_1, (1152, ), (1, ))
    assert_size_stride(arg74_1, (1152, ), (1, ))
    assert_size_stride(arg75_1, (1152, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (1152, ), (1, ))
    assert_size_stride(arg79_1, (1152, ), (1, ))
    assert_size_stride(arg80_1, (1152, ), (1, ))
    assert_size_stride(arg81_1, (1152, ), (1, ))
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
    assert_size_stride(arg94_1, (320, ), (1, ))
    assert_size_stride(arg95_1, (320, ), (1, ))
    assert_size_stride(arg96_1, (1280, ), (1, ))
    assert_size_stride(arg97_1, (1280, ), (1, ))
    assert_size_stride(arg98_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg99_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg101_1, (8, ), (1, ))
    assert_size_stride(arg102_1, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg103_1, (32, ), (1, ))
    assert_size_stride(arg104_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg105_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg106_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg107_1, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg108_1, (4, ), (1, ))
    assert_size_stride(arg109_1, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg110_1, (96, ), (1, ))
    assert_size_stride(arg111_1, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg112_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg113_1, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg114_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg115_1, (6, ), (1, ))
    assert_size_stride(arg116_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg117_1, (144, ), (1, ))
    assert_size_stride(arg118_1, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg119_1, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg120_1, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg121_1, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg122_1, (6, ), (1, ))
    assert_size_stride(arg123_1, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(arg124_1, (144, ), (1, ))
    assert_size_stride(arg125_1, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg126_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg127_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg128_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg129_1, (10, ), (1, ))
    assert_size_stride(arg130_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg131_1, (240, ), (1, ))
    assert_size_stride(arg132_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg133_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg134_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg135_1, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg136_1, (10, ), (1, ))
    assert_size_stride(arg137_1, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(arg138_1, (240, ), (1, ))
    assert_size_stride(arg139_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg140_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg141_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg143_1, (20, ), (1, ))
    assert_size_stride(arg144_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg145_1, (480, ), (1, ))
    assert_size_stride(arg146_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg147_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg148_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg149_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg150_1, (20, ), (1, ))
    assert_size_stride(arg151_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg152_1, (480, ), (1, ))
    assert_size_stride(arg153_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg154_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg155_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg156_1, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg157_1, (20, ), (1, ))
    assert_size_stride(arg158_1, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg161_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg162_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg163_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg164_1, (28, ), (1, ))
    assert_size_stride(arg165_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg166_1, (672, ), (1, ))
    assert_size_stride(arg167_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg168_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg169_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg170_1, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg171_1, (28, ), (1, ))
    assert_size_stride(arg172_1, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg173_1, (672, ), (1, ))
    assert_size_stride(arg174_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg175_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg176_1, (672, 1, 5, 5), (25, 25, 5, 1))
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
    assert_size_stride(arg311_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg311_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg311_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg98_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg98_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 112, 112), (401408, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg213_1, arg214_1, arg0_1, arg1_1, buf4, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg0_1
        del arg1_1
        del arg213_1
        del arg214_1
        del buf3
        # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.silu]
        buf5 = extern_kernels.convolution(buf4, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (4, 32, 112, 112), (401408, 12544, 112, 1))
        del arg99_1
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6, x_9, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_3.run(buf6, arg215_1, arg216_1, arg2_1, arg3_1, buf7, 256, 6272, grid=grid(256), stream=stream0)
        del arg215_1
        del arg216_1
        del arg2_1
        del arg3_1
        buf8 = empty_strided((4, 32, 1, 1), (32, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf9 = reinterpret_tensor(buf8, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf8  # reuse
        # Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_4.run(buf9, buf7, 128, 2, grid=grid(128), stream=stream0)
        del buf7
        # Source Nodes: [x_9, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf10 = extern_kernels.convolution(buf9, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 8, 1, 1), (8, 1, 1, 1))
        del arg100_1
        del buf9
        buf11 = reinterpret_tensor(buf10, (4, 8, 1, 1), (8, 1, 8, 8), 0); del buf10  # reuse
        # Source Nodes: [x_9, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_5.run(buf11, arg101_1, 32, grid=grid(32), stream=stream0)
        del arg101_1
        # Source Nodes: [x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf12 = extern_kernels.convolution(buf11, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 32, 1, 1), (32, 1, 1, 1))
        del arg102_1
        del buf11
        buf13 = buf4; del buf4  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_6.run(buf6, buf12, arg103_1, buf13, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg103_1
        del buf12
        del buf6
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_11, x_9, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf14 = extern_kernels.convolution(buf13, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 16, 112, 112), (200704, 12544, 112, 1))
        del arg104_1
        del buf13
        buf15 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf14, arg217_1, arg218_1, arg4_1, arg5_1, buf15, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg217_1
        del arg218_1
        del arg4_1
        del arg5_1
        del buf14
        # Source Nodes: [x_12, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 96, 112, 112), (1204224, 12544, 112, 1))
        del arg105_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        buf18 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf17, arg219_1, arg220_1, arg6_1, arg7_1, buf18, 384, 12544, grid=grid(384, 12544), stream=stream0)
        del arg219_1
        del arg220_1
        del arg6_1
        del arg7_1
        del buf17
        # Source Nodes: [x_20, x_21], Original ATen: [aten.convolution, aten.silu]
        buf19 = extern_kernels.convolution(buf18, arg106_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf19, (4, 96, 56, 56), (301056, 3136, 56, 1))
        del arg106_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        buf21 = empty_strided((4, 96, 1, 1), (96, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf22 = reinterpret_tensor(buf21, (4, 96, 1, 1), (96, 1, 96, 96), 0); del buf21  # reuse
        # Source Nodes: [x_22, x_25, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_9.run(buf20, buf22, arg221_1, arg222_1, arg8_1, arg9_1, 384, 3136, grid=grid(384), stream=stream0)
        del arg221_1
        del arg222_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_25, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf23 = extern_kernels.convolution(buf22, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 4, 1, 1), (4, 1, 1, 1))
        del arg107_1
        del buf22
        buf24 = reinterpret_tensor(buf23, (4, 4, 1, 1), (4, 1, 4, 4), 0); del buf23  # reuse
        # Source Nodes: [x_25, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_10.run(buf24, arg108_1, 16, grid=grid(16), stream=stream0)
        del arg108_1
        # Source Nodes: [x_25, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf25 = extern_kernels.convolution(buf24, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 96, 1, 1), (96, 1, 1, 1))
        del arg109_1
        del buf24
        buf26 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_11.run(buf20, buf25, arg110_1, buf26, 384, 3136, grid=grid(384, 3136), stream=stream0)
        del arg110_1
        del buf20
        del buf25
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26, x_27, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf27 = extern_kernels.convolution(buf26, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg111_1
        del buf26
        buf28 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf27, arg223_1, arg224_1, arg10_1, arg11_1, buf28, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg223_1
        del arg224_1
        del buf27
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 144, 56, 56), (451584, 3136, 56, 1))
        del arg112_1
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf30, arg225_1, arg226_1, arg12_1, arg13_1, buf31, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg225_1
        del arg226_1
        del buf30
        # Source Nodes: [x_36, x_37], Original ATen: [aten.convolution, aten.silu]
        buf32 = extern_kernels.convolution(buf31, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf32, (4, 144, 56, 56), (451584, 3136, 56, 1))
        del arg113_1
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided((4, 144, 1, 1), (144, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf35 = reinterpret_tensor(buf34, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf34  # reuse
        # Source Nodes: [x_38, x_41, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_red_fused__native_batch_norm_legit_no_training_mean_silu_14.run(buf33, buf35, arg227_1, arg228_1, arg14_1, arg15_1, 576, 3136, grid=grid(576), stream=stream0)
        del arg14_1
        del arg15_1
        del arg227_1
        del arg228_1
        # Source Nodes: [x_41, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf36 = extern_kernels.convolution(buf35, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 6, 1, 1), (6, 1, 1, 1))
        del arg114_1
        del buf35
        buf37 = reinterpret_tensor(buf36, (4, 6, 1, 1), (6, 1, 6, 6), 0); del buf36  # reuse
        # Source Nodes: [x_41, x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_15.run(buf37, arg115_1, 24, grid=grid(24), stream=stream0)
        del arg115_1
        # Source Nodes: [x_41, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf38 = extern_kernels.convolution(buf37, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 144, 1, 1), (144, 1, 1, 1))
        del arg116_1
        del buf37
        buf39 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_16.run(buf33, buf38, arg117_1, buf39, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg117_1
        del buf33
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42, x_43, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf40 = extern_kernels.convolution(buf39, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg118_1
        buf41 = buf28; del buf28  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf41, buf40, arg229_1, arg230_1, arg16_1, arg17_1, 12544, 24, grid=grid(12544, 24), stream=stream0)
        del arg16_1
        del arg17_1
        del arg229_1
        del arg230_1
        del buf40
        # Source Nodes: [shortcut_3, x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 144, 56, 56), (451584, 3136, 56, 1))
        del arg119_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        buf44 = buf39; del buf39  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf43, arg231_1, arg232_1, arg18_1, arg19_1, buf44, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg18_1
        del arg19_1
        del arg231_1
        del arg232_1
        del buf43
        # Source Nodes: [x_53, x_54], Original ATen: [aten.convolution, aten.silu]
        buf45 = extern_kernels.convolution(buf44, arg120_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf45, (4, 144, 28, 28), (112896, 784, 28, 1))
        del arg120_1
        del buf44
        buf46 = buf45; del buf45  # reuse
        buf47 = reinterpret_tensor(buf38, (4, 144, 1, 1), (144, 1, 576, 576), 0); del buf38  # reuse
        buf48 = reinterpret_tensor(buf47, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf47  # reuse
        # Source Nodes: [x_55, x_58, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_18.run(buf46, buf48, arg233_1, arg234_1, arg20_1, arg21_1, 576, 784, grid=grid(576), stream=stream0)
        del arg20_1
        del arg21_1
        del arg233_1
        del arg234_1
        # Source Nodes: [x_58, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf49 = extern_kernels.convolution(buf48, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 6, 1, 1), (6, 1, 1, 1))
        del arg121_1
        del buf48
        buf50 = reinterpret_tensor(buf49, (4, 6, 1, 1), (6, 1, 6, 6), 0); del buf49  # reuse
        # Source Nodes: [x_58, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_15.run(buf50, arg122_1, 24, grid=grid(24), stream=stream0)
        del arg122_1
        # Source Nodes: [x_58, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf51 = extern_kernels.convolution(buf50, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 144, 1, 1), (144, 1, 1, 1))
        del arg123_1
        del buf50
        buf52 = empty_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_19.run(buf46, buf51, arg124_1, buf52, 576, 784, grid=grid(576, 784), stream=stream0)
        del arg124_1
        del buf46
        del buf51
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59, x_60, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf53 = extern_kernels.convolution(buf52, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg125_1
        del buf52
        buf54 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf53, arg235_1, arg236_1, arg22_1, arg23_1, buf54, 160, 784, grid=grid(160, 784), stream=stream0)
        del arg22_1
        del arg235_1
        del arg236_1
        del arg23_1
        del buf53
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 240, 28, 28), (188160, 784, 28, 1))
        del arg126_1
        buf56 = buf55; del buf55  # reuse
        buf57 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_21.run(buf56, arg237_1, arg238_1, arg24_1, arg25_1, buf57, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg237_1
        del arg238_1
        del arg24_1
        del arg25_1
        del buf56
        # Source Nodes: [x_69, x_70], Original ATen: [aten.convolution, aten.silu]
        buf58 = extern_kernels.convolution(buf57, arg127_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf58, (4, 240, 28, 28), (188160, 784, 28, 1))
        del arg127_1
        buf59 = buf58; del buf58  # reuse
        buf60 = empty_strided((4, 240, 1, 1), (240, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf61 = reinterpret_tensor(buf60, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf60  # reuse
        # Source Nodes: [x_71, x_74, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_22.run(buf59, buf61, arg239_1, arg240_1, arg26_1, arg27_1, 960, 784, grid=grid(960), stream=stream0)
        del arg239_1
        del arg240_1
        del arg26_1
        del arg27_1
        # Source Nodes: [x_74, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf62 = extern_kernels.convolution(buf61, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 10, 1, 1), (10, 1, 1, 1))
        del arg128_1
        del buf61
        buf63 = reinterpret_tensor(buf62, (4, 10, 1, 1), (10, 1, 10, 10), 0); del buf62  # reuse
        # Source Nodes: [x_74, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_23.run(buf63, arg129_1, 40, grid=grid(40), stream=stream0)
        del arg129_1
        # Source Nodes: [x_74, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf64 = extern_kernels.convolution(buf63, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 240, 1, 1), (240, 1, 1, 1))
        del arg130_1
        del buf63
        buf65 = buf57; del buf57  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_24.run(buf59, buf64, arg131_1, buf65, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg131_1
        del buf59
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75, x_76, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf66 = extern_kernels.convolution(buf65, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg132_1
        buf67 = buf54; del buf54  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_25.run(buf67, buf66, arg241_1, arg242_1, arg28_1, arg29_1, 3136, 40, grid=grid(3136, 40), stream=stream0)
        del arg241_1
        del arg242_1
        del arg28_1
        del arg29_1
        del buf66
        # Source Nodes: [shortcut_5, x_77, x_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 240, 28, 28), (188160, 784, 28, 1))
        del arg133_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        buf70 = buf65; del buf65  # reuse
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_21.run(buf69, arg243_1, arg244_1, arg30_1, arg31_1, buf70, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg243_1
        del arg244_1
        del arg30_1
        del arg31_1
        del buf69
        # Source Nodes: [x_86, x_87], Original ATen: [aten.convolution, aten.silu]
        buf71 = extern_kernels.convolution(buf70, arg134_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf71, (4, 240, 14, 14), (47040, 196, 14, 1))
        del arg134_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        buf73 = reinterpret_tensor(buf64, (4, 240, 1, 1), (240, 1, 960, 960), 0); del buf64  # reuse
        buf74 = reinterpret_tensor(buf73, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf73  # reuse
        # Source Nodes: [x_88, x_91, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_26.run(buf72, buf74, arg245_1, arg246_1, arg32_1, arg33_1, 960, 196, grid=grid(960), stream=stream0)
        del arg245_1
        del arg246_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_91, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf75 = extern_kernels.convolution(buf74, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 10, 1, 1), (10, 1, 1, 1))
        del arg135_1
        del buf74
        buf76 = reinterpret_tensor(buf75, (4, 10, 1, 1), (10, 1, 10, 10), 0); del buf75  # reuse
        # Source Nodes: [x_91, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_23.run(buf76, arg136_1, 40, grid=grid(40), stream=stream0)
        del arg136_1
        # Source Nodes: [x_91, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf77 = extern_kernels.convolution(buf76, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 240, 1, 1), (240, 1, 1, 1))
        del arg137_1
        del buf76
        buf78 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_27.run(buf72, buf77, arg138_1, buf78, 960, 196, grid=grid(960, 196), stream=stream0)
        del arg138_1
        del buf72
        del buf77
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92, x_93, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf79 = extern_kernels.convolution(buf78, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg139_1
        del buf78
        buf80 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf79, arg247_1, arg248_1, arg34_1, arg35_1, buf80, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg247_1
        del arg248_1
        del arg34_1
        del arg35_1
        del buf79
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg140_1
        buf82 = buf81; del buf81  # reuse
        buf83 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_29.run(buf82, arg249_1, arg250_1, arg36_1, arg37_1, buf83, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg249_1
        del arg250_1
        del arg36_1
        del arg37_1
        del buf82
        # Source Nodes: [x_102, x_103], Original ATen: [aten.convolution, aten.silu]
        buf84 = extern_kernels.convolution(buf83, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf84, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg141_1
        buf85 = buf84; del buf84  # reuse
        buf86 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf87 = reinterpret_tensor(buf86, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf86  # reuse
        # Source Nodes: [x_104, x_107, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30.run(buf85, buf87, arg251_1, arg252_1, arg38_1, arg39_1, 1920, 196, grid=grid(1920), stream=stream0)
        del arg251_1
        del arg252_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_107, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf88 = extern_kernels.convolution(buf87, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 20, 1, 1), (20, 1, 1, 1))
        del arg142_1
        del buf87
        buf89 = reinterpret_tensor(buf88, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf88  # reuse
        # Source Nodes: [x_107, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_31.run(buf89, arg143_1, 80, grid=grid(80), stream=stream0)
        del arg143_1
        # Source Nodes: [x_107, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf90 = extern_kernels.convolution(buf89, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 480, 1, 1), (480, 1, 1, 1))
        del arg144_1
        del buf89
        buf91 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_32.run(buf85, buf90, arg145_1, buf91, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg145_1
        del buf85
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108, x_109, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf92 = extern_kernels.convolution(buf91, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg146_1
        buf93 = buf80; del buf80  # reuse
        # Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_33.run(buf93, buf92, arg253_1, arg254_1, arg40_1, arg41_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg253_1
        del arg254_1
        del arg40_1
        del arg41_1
        del buf92
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg147_1
        buf95 = buf94; del buf94  # reuse
        buf96 = buf91; del buf91  # reuse
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_29.run(buf95, arg255_1, arg256_1, arg42_1, arg43_1, buf96, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg255_1
        del arg256_1
        del arg42_1
        del arg43_1
        del buf95
        # Source Nodes: [x_119, x_120], Original ATen: [aten.convolution, aten.silu]
        buf97 = extern_kernels.convolution(buf96, arg148_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf97, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg148_1
        buf98 = buf97; del buf97  # reuse
        buf99 = reinterpret_tensor(buf90, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf90  # reuse
        buf100 = reinterpret_tensor(buf99, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf99  # reuse
        # Source Nodes: [x_121, x_124, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30.run(buf98, buf100, arg257_1, arg258_1, arg44_1, arg45_1, 1920, 196, grid=grid(1920), stream=stream0)
        del arg257_1
        del arg258_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_124, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf101 = extern_kernels.convolution(buf100, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 20, 1, 1), (20, 1, 1, 1))
        del arg149_1
        del buf100
        buf102 = reinterpret_tensor(buf101, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf101  # reuse
        # Source Nodes: [x_124, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_31.run(buf102, arg150_1, 80, grid=grid(80), stream=stream0)
        del arg150_1
        # Source Nodes: [x_124, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf103 = extern_kernels.convolution(buf102, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 480, 1, 1), (480, 1, 1, 1))
        del arg151_1
        del buf102
        buf104 = buf96; del buf96  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_32.run(buf98, buf103, arg152_1, buf104, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg152_1
        del buf98
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125, x_126, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf105 = extern_kernels.convolution(buf104, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg153_1
        buf106 = buf93; del buf93  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_33.run(buf106, buf105, arg259_1, arg260_1, arg46_1, arg47_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg259_1
        del arg260_1
        del arg46_1
        del arg47_1
        del buf105
        # Source Nodes: [shortcut_8, x_127, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg154_1
        buf108 = buf107; del buf107  # reuse
        buf109 = buf104; del buf104  # reuse
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_29.run(buf108, arg261_1, arg262_1, arg48_1, arg49_1, buf109, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg261_1
        del arg262_1
        del arg48_1
        del arg49_1
        del buf108
        # Source Nodes: [x_136, x_137], Original ATen: [aten.convolution, aten.silu]
        buf110 = extern_kernels.convolution(buf109, arg155_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf110, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg155_1
        buf111 = buf110; del buf110  # reuse
        buf112 = reinterpret_tensor(buf103, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf103  # reuse
        buf113 = reinterpret_tensor(buf112, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf112  # reuse
        # Source Nodes: [x_138, x_141, x_se_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_30.run(buf111, buf113, arg263_1, arg264_1, arg50_1, arg51_1, 1920, 196, grid=grid(1920), stream=stream0)
        del arg263_1
        del arg264_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_141, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf114 = extern_kernels.convolution(buf113, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 20, 1, 1), (20, 1, 1, 1))
        del arg156_1
        del buf113
        buf115 = reinterpret_tensor(buf114, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf114  # reuse
        # Source Nodes: [x_141, x_se_32, x_se_33, x_se_34], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_31.run(buf115, arg157_1, 80, grid=grid(80), stream=stream0)
        del arg157_1
        # Source Nodes: [x_141, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf116 = extern_kernels.convolution(buf115, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 480, 1, 1), (480, 1, 1, 1))
        del arg158_1
        del buf115
        buf117 = buf109; del buf109  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_141, x_142, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_32.run(buf111, buf116, arg159_1, buf117, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg159_1
        del buf111
        del buf116
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_141, x_142, x_143, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf118 = extern_kernels.convolution(buf117, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 112, 14, 14), (21952, 196, 14, 1))
        del arg160_1
        del buf117
        buf119 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf118, arg265_1, arg266_1, arg52_1, arg53_1, buf119, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg265_1
        del arg266_1
        del arg52_1
        del arg53_1
        del buf118
        # Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg161_1
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_35.run(buf121, arg267_1, arg268_1, arg54_1, arg55_1, buf122, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg267_1
        del arg268_1
        del arg54_1
        del arg55_1
        del buf121
        # Source Nodes: [x_152, x_153], Original ATen: [aten.convolution, aten.silu]
        buf123 = extern_kernels.convolution(buf122, arg162_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf123, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg162_1
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf126 = reinterpret_tensor(buf125, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf125  # reuse
        # Source Nodes: [x_154, x_157, x_se_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_36.run(buf124, buf126, arg269_1, arg270_1, arg56_1, arg57_1, 2688, 196, grid=grid(2688), stream=stream0)
        del arg269_1
        del arg270_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_157, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf127 = extern_kernels.convolution(buf126, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 28, 1, 1), (28, 1, 1, 1))
        del arg163_1
        del buf126
        buf128 = reinterpret_tensor(buf127, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf127  # reuse
        # Source Nodes: [x_157, x_se_36, x_se_37, x_se_38], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_37.run(buf128, arg164_1, 112, grid=grid(112), stream=stream0)
        del arg164_1
        # Source Nodes: [x_157, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf129 = extern_kernels.convolution(buf128, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 672, 1, 1), (672, 1, 1, 1))
        del arg165_1
        del buf128
        buf130 = buf122; del buf122  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_157, x_158, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_38.run(buf124, buf129, arg166_1, buf130, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg166_1
        del buf124
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_157, x_158, x_159, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf131 = extern_kernels.convolution(buf130, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 112, 14, 14), (21952, 196, 14, 1))
        del arg167_1
        buf132 = buf119; del buf119  # reuse
        # Source Nodes: [shortcut_10, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_39.run(buf132, buf131, arg271_1, arg272_1, arg58_1, arg59_1, 784, 112, grid=grid(784, 112), stream=stream0)
        del arg271_1
        del arg272_1
        del arg58_1
        del arg59_1
        del buf131
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg168_1
        buf134 = buf133; del buf133  # reuse
        buf135 = buf130; del buf130  # reuse
        # Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_35.run(buf134, arg273_1, arg274_1, arg60_1, arg61_1, buf135, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg273_1
        del arg274_1
        del arg60_1
        del arg61_1
        del buf134
        # Source Nodes: [x_169, x_170], Original ATen: [aten.convolution, aten.silu]
        buf136 = extern_kernels.convolution(buf135, arg169_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf136, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg169_1
        buf137 = buf136; del buf136  # reuse
        buf138 = reinterpret_tensor(buf129, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf129  # reuse
        buf139 = reinterpret_tensor(buf138, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf138  # reuse
        # Source Nodes: [x_171, x_174, x_se_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_36.run(buf137, buf139, arg275_1, arg276_1, arg62_1, arg63_1, 2688, 196, grid=grid(2688), stream=stream0)
        del arg275_1
        del arg276_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_174, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf140 = extern_kernels.convolution(buf139, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 28, 1, 1), (28, 1, 1, 1))
        del arg170_1
        del buf139
        buf141 = reinterpret_tensor(buf140, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf140  # reuse
        # Source Nodes: [x_174, x_se_40, x_se_41, x_se_42], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_37.run(buf141, arg171_1, 112, grid=grid(112), stream=stream0)
        del arg171_1
        # Source Nodes: [x_174, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf142 = extern_kernels.convolution(buf141, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 672, 1, 1), (672, 1, 1, 1))
        del arg172_1
        del buf141
        buf143 = buf135; del buf135  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_174, x_175, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_38.run(buf137, buf142, arg173_1, buf143, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg173_1
        del buf137
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_174, x_175, x_176, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf144 = extern_kernels.convolution(buf143, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 112, 14, 14), (21952, 196, 14, 1))
        del arg174_1
        buf145 = buf132; del buf132  # reuse
        # Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_39.run(buf145, buf144, arg277_1, arg278_1, arg64_1, arg65_1, 784, 112, grid=grid(784, 112), stream=stream0)
        del arg277_1
        del arg278_1
        del arg64_1
        del arg65_1
        del buf144
        # Source Nodes: [shortcut_11, x_177, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg175_1
        del buf145
        buf147 = buf146; del buf146  # reuse
        buf148 = buf143; del buf143  # reuse
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_35.run(buf147, arg279_1, arg280_1, arg66_1, arg67_1, buf148, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg279_1
        del arg280_1
        del arg66_1
        del arg67_1
        del buf147
        # Source Nodes: [x_186, x_187], Original ATen: [aten.convolution, aten.silu]
        buf149 = extern_kernels.convolution(buf148, arg176_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf149, (4, 672, 7, 7), (32928, 49, 7, 1))
        del arg176_1
        del buf148
        buf150 = buf149; del buf149  # reuse
        buf151 = reinterpret_tensor(buf142, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf142  # reuse
        buf152 = reinterpret_tensor(buf151, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf151  # reuse
        # Source Nodes: [x_188, x_191, x_se_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_40.run(buf150, buf152, arg281_1, arg282_1, arg68_1, arg69_1, 2688, 49, grid=grid(2688), stream=stream0)
        del arg281_1
        del arg282_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_191, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf153 = extern_kernels.convolution(buf152, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (4, 28, 1, 1), (28, 1, 1, 1))
        del arg177_1
        del buf152
        buf154 = reinterpret_tensor(buf153, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf153  # reuse
        # Source Nodes: [x_191, x_se_44, x_se_45, x_se_46], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_37.run(buf154, arg178_1, 112, grid=grid(112), stream=stream0)
        del arg178_1
        # Source Nodes: [x_191, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf155 = extern_kernels.convolution(buf154, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 672, 1, 1), (672, 1, 1, 1))
        del arg179_1
        del buf154
        buf156 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_191, x_192, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_41.run(buf150, buf155, arg180_1, buf156, 2688, 49, grid=grid(2688, 49), stream=stream0)
        del arg180_1
        del buf150
        del buf155
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_191, x_192, x_193, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf157 = extern_kernels.convolution(buf156, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg181_1
        del buf156
        buf158 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf157, arg283_1, arg284_1, arg70_1, arg71_1, buf158, 768, 49, grid=grid(768, 49), stream=stream0)
        del arg283_1
        del arg284_1
        del arg70_1
        del arg71_1
        del buf157
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg182_1
        buf160 = buf159; del buf159  # reuse
        buf161 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf160, arg285_1, arg286_1, arg72_1, arg73_1, buf161, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg285_1
        del arg286_1
        del arg72_1
        del arg73_1
        del buf160
        # Source Nodes: [x_202, x_203], Original ATen: [aten.convolution, aten.silu]
        buf162 = extern_kernels.convolution(buf161, arg183_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf162, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg183_1
        buf163 = buf162; del buf162  # reuse
        buf164 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cuda', dtype=torch.float32)
        buf165 = reinterpret_tensor(buf164, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf164  # reuse
        # Source Nodes: [x_204, x_207, x_se_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44.run(buf163, buf165, arg287_1, arg288_1, arg74_1, arg75_1, 4608, 49, grid=grid(4608), stream=stream0)
        del arg287_1
        del arg288_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_207, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf166 = extern_kernels.convolution(buf165, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 48, 1, 1), (48, 1, 1, 1))
        del arg184_1
        del buf165
        buf167 = reinterpret_tensor(buf166, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf166  # reuse
        # Source Nodes: [x_207, x_se_48, x_se_49, x_se_50], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_45.run(buf167, arg185_1, 192, grid=grid(192), stream=stream0)
        del arg185_1
        # Source Nodes: [x_207, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf168 = extern_kernels.convolution(buf167, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 1152, 1, 1), (1152, 1, 1, 1))
        del arg186_1
        del buf167
        buf169 = buf161; del buf161  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_207, x_208, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_46.run(buf163, buf168, arg187_1, buf169, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg187_1
        del buf163
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_207, x_208, x_209, x_se_48, x_se_49, x_se_50, x_se_51], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf170 = extern_kernels.convolution(buf169, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg188_1
        buf171 = buf158; del buf158  # reuse
        # Source Nodes: [shortcut_13, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf171, buf170, arg289_1, arg290_1, arg76_1, arg77_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg289_1
        del arg290_1
        del arg76_1
        del arg77_1
        del buf170
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg189_1
        buf173 = buf172; del buf172  # reuse
        buf174 = buf169; del buf169  # reuse
        # Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf173, arg291_1, arg292_1, arg78_1, arg79_1, buf174, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg291_1
        del arg292_1
        del arg78_1
        del arg79_1
        del buf173
        # Source Nodes: [x_219, x_220], Original ATen: [aten.convolution, aten.silu]
        buf175 = extern_kernels.convolution(buf174, arg190_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf175, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg190_1
        buf176 = buf175; del buf175  # reuse
        buf177 = reinterpret_tensor(buf168, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf168  # reuse
        buf178 = reinterpret_tensor(buf177, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf177  # reuse
        # Source Nodes: [x_221, x_224, x_se_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44.run(buf176, buf178, arg293_1, arg294_1, arg80_1, arg81_1, 4608, 49, grid=grid(4608), stream=stream0)
        del arg293_1
        del arg294_1
        del arg80_1
        del arg81_1
        # Source Nodes: [x_224, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf179 = extern_kernels.convolution(buf178, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 48, 1, 1), (48, 1, 1, 1))
        del arg191_1
        del buf178
        buf180 = reinterpret_tensor(buf179, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf179  # reuse
        # Source Nodes: [x_224, x_se_52, x_se_53, x_se_54], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_45.run(buf180, arg192_1, 192, grid=grid(192), stream=stream0)
        del arg192_1
        # Source Nodes: [x_224, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf181 = extern_kernels.convolution(buf180, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 1152, 1, 1), (1152, 1, 1, 1))
        del arg193_1
        del buf180
        buf182 = buf174; del buf174  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_224, x_225, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_46.run(buf176, buf181, arg194_1, buf182, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg194_1
        del buf176
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_224, x_225, x_226, x_se_52, x_se_53, x_se_54, x_se_55], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf183 = extern_kernels.convolution(buf182, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg195_1
        buf184 = buf171; del buf171  # reuse
        # Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf184, buf183, arg295_1, arg296_1, arg82_1, arg83_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg295_1
        del arg296_1
        del arg82_1
        del arg83_1
        del buf183
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg196_1
        buf186 = buf185; del buf185  # reuse
        buf187 = buf182; del buf182  # reuse
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf186, arg297_1, arg298_1, arg84_1, arg85_1, buf187, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg297_1
        del arg298_1
        del arg84_1
        del arg85_1
        del buf186
        # Source Nodes: [x_236, x_237], Original ATen: [aten.convolution, aten.silu]
        buf188 = extern_kernels.convolution(buf187, arg197_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf188, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg197_1
        buf189 = buf188; del buf188  # reuse
        buf190 = reinterpret_tensor(buf181, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf181  # reuse
        buf191 = reinterpret_tensor(buf190, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf190  # reuse
        # Source Nodes: [x_238, x_241, x_se_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44.run(buf189, buf191, arg299_1, arg300_1, arg86_1, arg87_1, 4608, 49, grid=grid(4608), stream=stream0)
        del arg299_1
        del arg300_1
        del arg86_1
        del arg87_1
        # Source Nodes: [x_241, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf192 = extern_kernels.convolution(buf191, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 48, 1, 1), (48, 1, 1, 1))
        del arg198_1
        del buf191
        buf193 = reinterpret_tensor(buf192, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf192  # reuse
        # Source Nodes: [x_241, x_se_56, x_se_57, x_se_58], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_45.run(buf193, arg199_1, 192, grid=grid(192), stream=stream0)
        del arg199_1
        # Source Nodes: [x_241, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf194 = extern_kernels.convolution(buf193, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 1152, 1, 1), (1152, 1, 1, 1))
        del arg200_1
        del buf193
        buf195 = buf187; del buf187  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_241, x_242, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_46.run(buf189, buf194, arg201_1, buf195, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg201_1
        del buf189
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_241, x_242, x_243, x_se_56, x_se_57, x_se_58, x_se_59], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf196 = extern_kernels.convolution(buf195, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg202_1
        buf197 = buf184; del buf184  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf197, buf196, arg301_1, arg302_1, arg88_1, arg89_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg301_1
        del arg302_1
        del arg88_1
        del arg89_1
        del buf196
        # Source Nodes: [shortcut_15, x_244, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg203_1
        del buf197
        buf199 = buf198; del buf198  # reuse
        buf200 = buf195; del buf195  # reuse
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_43.run(buf199, arg303_1, arg304_1, arg90_1, arg91_1, buf200, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg303_1
        del arg304_1
        del arg90_1
        del arg91_1
        del buf199
        # Source Nodes: [x_253, x_254], Original ATen: [aten.convolution, aten.silu]
        buf201 = extern_kernels.convolution(buf200, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf201, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg204_1
        buf202 = buf201; del buf201  # reuse
        buf203 = reinterpret_tensor(buf194, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf194  # reuse
        buf204 = reinterpret_tensor(buf203, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf203  # reuse
        # Source Nodes: [x_255, x_258, x_se_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_44.run(buf202, buf204, arg305_1, arg306_1, arg92_1, arg93_1, 4608, 49, grid=grid(4608), stream=stream0)
        del arg305_1
        del arg306_1
        del arg92_1
        del arg93_1
        # Source Nodes: [x_258, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf205 = extern_kernels.convolution(buf204, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 48, 1, 1), (48, 1, 1, 1))
        del arg205_1
        del buf204
        buf206 = reinterpret_tensor(buf205, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf205  # reuse
        # Source Nodes: [x_258, x_se_60, x_se_61, x_se_62], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_poi_fused_convolution_mean_silu_45.run(buf206, arg206_1, 192, grid=grid(192), stream=stream0)
        del arg206_1
        # Source Nodes: [x_258, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf207 = extern_kernels.convolution(buf206, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 1152, 1, 1), (1152, 1, 1, 1))
        del arg207_1
        del buf206
        buf208 = buf200; del buf200  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_258, x_259, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_convolution_mean_mul_sigmoid_silu_46.run(buf202, buf207, arg208_1, buf208, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg208_1
        del buf202
        del buf207
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_258, x_259, x_260, x_se_60, x_se_61, x_se_62, x_se_63], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.sigmoid, aten.silu]
        buf209 = extern_kernels.convolution(buf208, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 320, 7, 7), (15680, 49, 7, 1))
        del arg209_1
        del buf208
        buf210 = reinterpret_tensor(buf106, (4, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf106  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf209, arg307_1, arg308_1, arg94_1, arg95_1, buf210, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del arg307_1
        del arg308_1
        del arg94_1
        del arg95_1
        del buf209
        # Source Nodes: [x_261, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 1280, 7, 7), (62720, 49, 7, 1))
        del arg210_1
        del buf210
        buf212 = buf211; del buf211  # reuse
        buf213 = empty_strided((4, 1280, 1, 1), (1280, 1, 5120, 5120), device='cuda', dtype=torch.float32)
        buf214 = reinterpret_tensor(buf213, (4, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf213  # reuse
        # Source Nodes: [x_267, x_271, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_silu_49.run(buf212, buf214, arg309_1, arg310_1, arg96_1, arg97_1, 5120, 49, grid=grid(5120), stream=stream0)
        del arg309_1
        del arg310_1
        del arg96_1
        del arg97_1
        del buf212
        buf215 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg212_1, reinterpret_tensor(buf214, (4, 1280), (1280, 1), 0), reinterpret_tensor(arg211_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf215)
        del arg211_1
        del arg212_1
        return (buf215, )


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
    arg52_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg70_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg94_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
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
    arg311_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_efficientnet', benchmark_compiled_module)
