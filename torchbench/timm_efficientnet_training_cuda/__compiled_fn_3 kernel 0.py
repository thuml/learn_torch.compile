
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7df2wzqzj65kpbrw3kvwd46dfszadkdpcxhoozudelgek5bfog5.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rp/crpyejihhghoq2yx2xrqpkpbbluhixyr7ir7p4imx6ilwsdv7bcr.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cddiafui2aenk2k6pegmmv3edykl5nk22qpi5k6jitm6skipivnd.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3myrhnff5oedut4bcaik2fxjtgdteby3zaznkqv6mv7etbd4y6v.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut => mul_3, sigmoid
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbnvy5blmz45a7vgszorn6ngnvp6bt6h67pqnq5pxxc47n2yrp4.py
# Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_6 => add_3, mul_5, mul_6, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevyqdze4e47fqcmjdmabfvp5dtgyrvx7hevowgsimcxucmoph4e.py
# Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
# x_9 => mul_7, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpoziczdhm7pivquqzv3r4qw7dudp5bavrgbyinsji56ywsropi.py
# Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
# x_9 => mul_7, sigmoid_1
# x_se => mean
triton_red_fused_mean_silu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_6', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (3136*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 12544.0
    tmp5 = tmp2 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckukjbhxdzg6iem4v2cgfjfwlcmlael2jcuuqiibozjzd47ai2pc.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
# x_se_1 => convolution_2
# x_se_2 => mul_8, sigmoid_2
triton_poi_fused_convolution_silu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzvu3whcba2xivorjjr6ue5ckwmdw6sd4dltcvgmnzproelgszf.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_3
triton_poi_fused_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlqr72cbscsme4qiu7xaxq2n7apyavsrg2gwyi2wiyilzfbwiaj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___0_____0___se_gate => sigmoid_3
# x_10 => mul_9
# x_9 => mul_7, sigmoid_1
triton_poi_fused_mul_sigmoid_silu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    x2 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (32*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4z6r6mkmzwpjdh6wxfz5cifbtpvt6gwjsjzd7aihlsrfz7t54l.py
# Source Nodes: [x_11], Original ATen: [aten.convolution]
# x_11 => convolution_4
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v3/cv347vakoeiaxowodnph6i3d6sifl3ujljkcxo7f2qonwknqgoz5.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_12 => add_5, mul_11, mul_12, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qk/cqka76ftwodwxovoio4axyejvkoml6siwksa36kkrerjf36zp2r4.py
# Source Nodes: [x_16], Original ATen: [aten.convolution]
# x_16 => convolution_5
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwy4u56qjfjvbja4xfqicqjnuqfcaanvtwmbtr5xjxlohn2t3ns.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_17 => add_7, mul_14, mul_15, sub_3
# x_20 => mul_16, sigmoid_4
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr74mzqh54va5cwmqgfmmlspbhc5keymi32u2ugxysnxtmnv72jw.py
# Source Nodes: [x_21], Original ATen: [aten.convolution]
# x_21 => convolution_6
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (301056*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykh4fr4h7u2k3ahe6ldzvqrhzaigeinryodqtj5fwjhf6c76hdd.py
# Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_22 => add_9, mul_18, mul_19, sub_4
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w4mnkfphns2ppptpidldsphi6xl5skc6qo4ofruhudt524v37g.py
# Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_25 => mul_20, sigmoid_5
# x_se_4 => mean_1
triton_red_fused_mean_silu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9600
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 96) % 25
    x0 = xindex % 96
    x2 = (xindex // 2400)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (96*((r3 + (126*x1)) % 3136)) + (301056*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfkmrprmcamw5nzt4kt64goaxhszrsns2jyblomsc5byieos4xx.py
# Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
# x_25 => mul_20, sigmoid_5
# x_se_4 => mean_1
triton_per_fused_mean_silu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 96
    x1 = (xindex // 96)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (2400*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sf/csf7qs4p2z7a2kuzg4j6zbp3wnmz2koz7pbixgnlrcdrzc5ttt2l.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
# x_se_5 => convolution_7
# x_se_6 => mul_21, sigmoid_6
triton_poi_fused_convolution_silu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdm3xmpoh3ihur3kdop7w65rcgzqnk6p7wfsu5ju24okllfpmby.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_8
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
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nvzmwgctd3azr4he4tdvhjyu3fefrg7dykdjcn2qtxmszpru35.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____0___se_gate => sigmoid_7
# x_25 => mul_20, sigmoid_5
# x_26 => mul_22
triton_poi_fused_mul_sigmoid_silu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 96
    x2 = (xindex // 301056)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (96*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjxl7olbjnpw72qf7if5fmh53dmundnlko4le6tsoq3qmkicjn2.py
# Source Nodes: [x_27], Original ATen: [aten.convolution]
# x_27 => convolution_9
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaddzytxqpxp6vqsf45jqrd7yvyady43ssrt6gfh262sks5xq57.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_28 => add_11, mul_24, mul_25, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxohqyo7dip37tgedr6l2ifjnmk6iar63c3tfm52z664zrgyn7y.py
# Source Nodes: [x_32], Original ATen: [aten.convolution]
# x_32 => convolution_10
triton_poi_fused_convolution_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (144*x2) + (451584*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64jlfsa543enizxoajt2pd6ouv3aqc6jqr4o6etfezsoub4le3a.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_33 => add_13, mul_27, mul_28, sub_6
# x_36 => mul_29, sigmoid_8
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, None)
    tl.store(out_ptr2 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csykloskwtodbj2frxtfxym7fist5xb2bqqiglbtakt6cpg5ewx2.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_38 => add_15, mul_31, mul_32, sub_7
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjshqujfe3ivbogf2wewgg2bhzqkex4kgtegrpdflp4eyaymijt2.py
# Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_41 => mul_33, sigmoid_9
# x_se_8 => mean_2
triton_red_fused_mean_silu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14400
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 144) % 25
    x0 = xindex % 144
    x2 = (xindex // 3600)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (144*((r3 + (126*x1)) % 3136)) + (451584*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.sigmoid(tmp3)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dnb5oj7tkkqymy5fpig3v3bdvk5w4ofyc5n3x662exowyoo3ll.py
# Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
# x_41 => mul_33, sigmoid_9
# x_se_8 => mean_2
triton_per_fused_mean_silu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (3600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 3136.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuvn5j373vv7r6hvgflvoboaahf3vh23dw3y6b3kkgtp4f4pfhi.py
# Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
# x_se_10 => mul_34, sigmoid_10
# x_se_9 => convolution_12
triton_poi_fused_convolution_silu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqnkv2qvowc46iyz7eq3fiax3jig5uyd4pc2lcyff6b4xjfpusd.py
# Source Nodes: [x_se_11], Original ATen: [aten.convolution]
# x_se_11 => convolution_13
triton_poi_fused_convolution_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmq6mquxjepufobjr76u2qudhpoxa2w7ymvfkqq3foyb7qw2ikz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___1_____1___se_gate => sigmoid_11
# x_41 => mul_33, sigmoid_9
# x_42 => mul_35
triton_poi_fused_mul_sigmoid_silu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 451584)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdmnfji3egytupiqnj62xeiekrjuko26j3xlowsqvzlhclddg6n.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_44 => add_17, mul_37, mul_38, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(out_ptr0 + (x2), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chs7wbfgw7ggjn3a3zcgwr2st3vg4e73fg5mqppavtspjlg7dmwk.py
# Source Nodes: [x_54], Original ATen: [aten.convolution]
# x_54 => convolution_16
triton_poi_fused_convolution_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (144*x2) + (112896*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dqyhk5ht4y2xjg4wl6jbjjumofonvel25cz6cefbf2wfipz6vc.py
# Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_55 => add_22, mul_44, mul_45, sub_10
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caotyf7ivkerzh2dgl6of2urd3xr7d6rlho7g6hpfq7pqlyqnql3.py
# Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_58 => mul_46, sigmoid_13
# x_se_12 => mean_3
triton_red_fused_mean_silu_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4032
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 144
    x1 = (xindex // 144)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (16128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthhd6p5rq7xsy3d3wrkjzhgi2nayqovb7yivde3mmuwdyuhk4h3.py
# Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
# x_58 => mul_46, sigmoid_13
# x_se_12 => mean_3
triton_per_fused_mean_silu_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (144*r2) + (1008*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgepr6xviwwr5tqjkv2cvrasxd5tujk26d4cyunbvb3xssgty76t.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_15
# x_58 => mul_46, sigmoid_13
# x_59 => mul_48
triton_poi_fused_mul_sigmoid_silu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 144
    x2 = (xindex // 112896)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (144*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevhfyqiyrocerknx3z2yfvacywa4s3uu7tf63gjs5y7ygtrjvwi.py
# Source Nodes: [x_60], Original ATen: [aten.convolution]
# x_60 => convolution_19
triton_poi_fused_convolution_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbnw44ucqbqdxfl4u5jrgpzqpvyoykp4qeieajyvpjn5dkzedks.py
# Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_61 => add_24, mul_50, mul_51, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwx4jeevrwadgktzz5fyoqmaopugvnj3ulgkfpvxoi46cxethgd5.py
# Source Nodes: [x_65], Original ATen: [aten.convolution]
# x_65 => convolution_20
triton_poi_fused_convolution_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ygup5s5j7gnzfc7sixkcdvwlas5p2rkqu3kv3h5dd3lljgi4lo.py
# Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_66 => add_26, mul_53, mul_54, sub_12
# x_69 => mul_55, sigmoid_16
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotwvvgtxo5vu3uyf4umzmwvspkp3zuzxoe3yqzay5nfajb7vxwk.py
# Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_71 => add_28, mul_57, mul_58, sub_13
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckemt6ls253ielay55rpckc3t57nqe2n2kes7d565uuwvlkaunf2.py
# Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
triton_red_fused_mean_silu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4jupfk6tchuqiacepwa3l7furzusaagtsdw5ump3igt7m5tsi2.py
# Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
# x_74 => mul_59, sigmoid_17
# x_se_16 => mean_4
triton_per_fused_mean_silu_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (1680*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 784.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutorh2dfrmtrih5fp7q2dorxvmxkdly24q4ar2fxhggtgdisj6i.py
# Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
# x_se_17 => convolution_22
# x_se_18 => mul_60, sigmoid_18
triton_poi_fused_convolution_silu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5u2mtdhggldeesv5qmp37zausbvmry26savyq4b3m7olmlt3pdc.py
# Source Nodes: [x_se_19], Original ATen: [aten.convolution]
# x_se_19 => convolution_23
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbs6sxquwio4khirm3a7kwtlqr3n4chn7erkio4wl4xd4ifz6y5r.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => sigmoid_19
# x_74 => mul_59, sigmoid_17
# x_75 => mul_61
triton_poi_fused_mul_sigmoid_silu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 188160)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccamxkqbbbh6ppqtp6ccwexvykgdjd7lvsdr6jzxarfiuunmy7a.py
# Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_31
# x_77 => add_30, mul_63, mul_64, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttjry3465utqoacrdtxvcfwkhzzte6ayq64qw4peo4dr4g2oemp.py
# Source Nodes: [x_87], Original ATen: [aten.convolution]
# x_87 => convolution_26
triton_poi_fused_convolution_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesei7m6mzgyiczgiftr5yeb5qf7eltdqmy7thyvzbg7kytfdcua.py
# Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_88 => add_35, mul_70, mul_71, sub_16
triton_poi_fused__native_batch_norm_legit_no_training_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncidc3bby6qn2gegbbyiufjyvuczkqw2ewosdlvbaq3rlqlodab.py
# Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_91 => mul_72, sigmoid_21
# x_se_20 => mean_5
triton_red_fused_mean_silu_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (23520*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuhhy2xqgk5qec2u47qd3saffr7phtm2qsywvxdtifxddba7udkz.py
# Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
# x_91 => mul_72, sigmoid_21
# x_se_20 => mean_5
triton_per_fused_mean_silu_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 240
    x1 = (xindex // 240)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (480*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gfsismk7afu3qbbfzzpj5lxwpuf7jpfpbs3zm3gjc6caxl7cer.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_23
# x_91 => mul_72, sigmoid_21
# x_92 => mul_74
triton_poi_fused_mul_sigmoid_silu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 240
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (240*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4nquh6mup4ykvdqcbvj3gggqswomin2u7maca4xhclpps4sgp4.py
# Source Nodes: [x_93], Original ATen: [aten.convolution]
# x_93 => convolution_29
triton_poi_fused_convolution_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofd7bwjy7sapbirp6d6v7b5e376wkjraxas5xlwcvvfub2m35xm.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_94 => add_37, mul_76, mul_77, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvnr4c4qs4po47hrxk6d5hdwn4hokulxropfbwoyx536zxrh5d5.py
# Source Nodes: [x_98], Original ATen: [aten.convolution]
# x_98 => convolution_30
triton_poi_fused_convolution_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvg6rqktg4oqxgqilpss2yfcb53mhyiuvgsh2vw2dlj7lcsoiqe.py
# Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_102 => mul_81, sigmoid_24
# x_99 => add_39, mul_79, mul_80, sub_18
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl26ofx3exnt4os3ppshcl4kuqjrvyhezfzqevbcoymgelu77all.py
# Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_104 => add_41, mul_83, mul_84, sub_19
triton_poi_fused__native_batch_norm_legit_no_training_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czculwpuddglnwevxxjics2x52za3rdpypa722ak23fhgbhumcy4.py
# Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
triton_red_fused_mean_silu_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (47040*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf26zwldc2pxvyfpj6bosfre633eqelilyrk7vjralhbeh2wpwt7.py
# Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
# x_107 => mul_85, sigmoid_25
# x_se_24 => mean_6
triton_per_fused_mean_silu_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_59', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 480
    x1 = (xindex // 480)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (960*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2b7n5xe7hdt3pgdtv46cjsvfiutcul6w54eqa3yl2yxbexsiso.py
# Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
# x_se_25 => convolution_32
# x_se_26 => mul_86, sigmoid_26
triton_poi_fused_convolution_silu_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casqsvfk3mudgnmq2th2k6fpar2m4bfbxnbszqxqpcr67u3g3ugh.py
# Source Nodes: [x_se_27], Original ATen: [aten.convolution]
# x_se_27 => convolution_33
triton_poi_fused_convolution_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7j/c7jf25hkk45xzvva54iqznz63fs4ippvlbrjnmbbj57dotjstgnp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___3_____1___se_gate => sigmoid_27
# x_107 => mul_85, sigmoid_25
# x_108 => mul_87
triton_poi_fused_mul_sigmoid_silu_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 480
    x2 = (xindex // 94080)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (480*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zgk7ubtq6y3rwnp74ao23isvh5y7kres2eurfazkqyy6ecgfl4.py
# Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_7 => add_44
# x_110 => add_43, mul_89, mul_90, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_add_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 80
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxd5cs5ghkcmc7cv4s56xkn2rwvc723qmtdvbd43762sosymggms.py
# Source Nodes: [x_143], Original ATen: [aten.convolution]
# x_143 => convolution_44
triton_poi_fused_convolution_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfalolncnqfptjrwyrnvhibwho4ca5eztal3hgn54xqhfymomn72.py
# Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_144 => add_57, mul_115, mul_116, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/cayswi7qkadqftfyktbemje5xyh5jnnhor24kquveiyc7j4lqljv.py
# Source Nodes: [x_148], Original ATen: [aten.convolution]
# x_148 => convolution_45
triton_poi_fused_convolution_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65ttdbxtw7hr7hbmcvw26bsvmhaqm3sj6eep2ogz2zdxqt43nk2.py
# Source Nodes: [x_149, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_149 => add_59, mul_118, mul_119, sub_27
# x_152 => mul_120, sigmoid_36
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwr26r3y3cmlkdwugm5mt44z3gktcinzoeottbq6zas5rgfrnki.py
# Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_154 => add_61, mul_122, mul_123, sub_28
triton_poi_fused__native_batch_norm_legit_no_training_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqprax7kmpn73zkipygjrfclnvafmbqwaf7vl7uqvd7vpe7fnc4u.py
# Source Nodes: [x_157, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_157 => mul_124, sigmoid_37
# x_se_36 => mean_9
triton_red_fused_mean_silu_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_silu_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy77kklbstaae57yergtcfap5ysq7d3xjefwpb7466sl4g5fw4qi.py
# Source Nodes: [x_157, x_se_36], Original ATen: [aten.mean, aten.silu]
# x_157 => mul_124, sigmoid_37
# x_se_36 => mean_9
triton_per_fused_mean_silu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_70', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (1344*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 196.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iep46fi2tzkcjozusfkj7eo5vruz6rrdniqyozxquwkmwo2ctf.py
# Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
# x_se_37 => convolution_47
# x_se_38 => mul_125, sigmoid_38
triton_poi_fused_convolution_silu_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3ocrvorloosc6xhpk3dugfefjfeiazb6nmryaomdimjlbxfzz2.py
# Source Nodes: [x_se_39], Original ATen: [aten.convolution]
# x_se_39 => convolution_48
triton_poi_fused_convolution_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocjop4hbap4qqhwq7vqi2tavo4tnyhzfaihkw6g3doxsstgheyw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_157, x_158], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => sigmoid_39
# x_157 => mul_124, sigmoid_37
# x_158 => mul_126
triton_poi_fused_mul_sigmoid_silu_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 131712)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbj2jda2wp6ceumt6t7vbhazveftmdbmdtz3vvnvon66mmtg2eq.py
# Source Nodes: [shortcut_10, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_10 => add_64
# x_160 => add_63, mul_128, mul_129, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_add_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 112
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3epm4w4er6ut67zrjw46pawgfwtnig7i6ktuk5o4fifv6q76fd.py
# Source Nodes: [x_187], Original ATen: [aten.convolution]
# x_187 => convolution_56
triton_poi_fused_convolution_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c444mnexvjwhksokiyt4caf2wucydkgevxrcnkc4trbt425g2fxd.py
# Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_188 => add_75, mul_148, mul_149, sub_34
triton_poi_fused__native_batch_norm_legit_no_training_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxesx2apz77fjfioajzuawtqiwanaydnmbjtfeoyzkyzectrcoip.py
# Source Nodes: [x_191, x_se_44], Original ATen: [aten.mean, aten.silu]
# x_191 => mul_150, sigmoid_45
# x_se_44 => mean_11
triton_per_fused_mean_silu_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgpokne42dkjajcicohge5blc6rh6upjbf6itmqvkep4hm3xolec.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_191, x_192], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_47
# x_191 => mul_150, sigmoid_45
# x_192 => mul_152
triton_poi_fused_mul_sigmoid_silu_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 672
    x2 = (xindex // 32928)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (672*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iy/ciyjy7gujnjehoxks766vdd2nnc22gzkgqfmy4nk5nnvr6weuwxr.py
# Source Nodes: [x_193], Original ATen: [aten.convolution]
# x_193 => convolution_59
triton_poi_fused_convolution_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmb6bilmcskgviea7rvlmw55zetnitobthh7urrp6qdmyvd34q4g.py
# Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_194 => add_77, mul_154, mul_155, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkxclt7raf64kptylflmwm6kjlwpp2e3mpfb246izal77ka4vwp.py
# Source Nodes: [x_198], Original ATen: [aten.convolution]
# x_198 => convolution_60
triton_poi_fused_convolution_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ytrtyfr67bjhlyy254zghzoxwlaxi2ecms4wcur2l5cxutupph.py
# Source Nodes: [x_199, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_199 => add_79, mul_157, mul_158, sub_36
# x_202 => mul_159, sigmoid_48
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = tmp8 - tmp15
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18 + tmp8
    tmp20 = tmp15 * tmp19
    tl.store(out_ptr1 + (x2), tmp16, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvmxlonui632zbxemxlptppts3cxpipoevnxy44cz7ax5ihlvpc.py
# Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_204 => add_81, mul_161, mul_162, sub_37
triton_poi_fused__native_batch_norm_legit_no_training_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chckiwbq34emr5fjpsvnqaa5xxrbkw5wnr3gf5zbl5aieryocbbo.py
# Source Nodes: [x_207, x_se_48], Original ATen: [aten.mean, aten.silu]
# x_207 => mul_163, sigmoid_49
# x_se_48 => mean_12
triton_per_fused_mean_silu_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_84', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1152
    x1 = (xindex // 1152)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r2) + (56448*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfpwum7dimtxpzw36vfnagqgbak52ak6uhzwpc3hppmwqa5dvxy.py
# Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
# x_se_49 => convolution_62
# x_se_50 => mul_164, sigmoid_50
triton_poi_fused_convolution_silu_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7argnuvly76r5u24etofm3l7cgllp76wllr5corehzajw6r4en.py
# Source Nodes: [x_se_51], Original ATen: [aten.convolution]
# x_se_51 => convolution_63
triton_poi_fused_convolution_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64rdxafv3yzeyaztwiwjohfyvp3xno4x3opk5j7e4pbp2zf3xv4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_207, x_208], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => sigmoid_51
# x_207 => mul_163, sigmoid_49
# x_208 => mul_165
triton_poi_fused_mul_sigmoid_silu_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_silu_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1152
    x2 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + (1152*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vs/cvsaorkgujn26pi2ktaxny7kiamxej2wt36l7gqo5avxd7qlivwd.py
# Source Nodes: [shortcut_13, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_13 => add_84
# x_210 => add_83, mul_167, mul_168, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_add_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnbuhzniitodlyvo2v3mxevnsojr7bo7jictmreb3ktc557jz37.py
# Source Nodes: [x_260], Original ATen: [aten.convolution]
# x_260 => convolution_79
triton_poi_fused_convolution_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwp6eq77j6tpe3doa5xzk2hn3gkxlxl7kunfnahto6nnugzmj3rm.py
# Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_261 => add_104, mul_206, mul_207, sub_47
triton_poi_fused__native_batch_norm_legit_no_training_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 320
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6avfoletrcbwcrzuxlwaytqiqu74uunrw6v7elbfu33d2y4oxtt.py
# Source Nodes: [x_266], Original ATen: [aten.convolution]
# x_266 => convolution_80
triton_poi_fused_convolution_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5120
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1280*x2) + (62720*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bz6u6mo6cnemq4wz3iisccl3c6qtzdnzflvwejjqsrgvjlj66i.py
# Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_267 => add_106, mul_209, mul_210, sub_48
triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_sub_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_sub_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tmp8 - tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp17 + tmp8
    tmp19 = tmp15 * tmp18
    tl.store(out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr1 + (x2), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2rzuizpbba3adr36fl73qu4uyylm6grcxfmhrolrmavfxntckkr.py
# Source Nodes: [x_271, x_272, x_274], Original ATen: [aten.mean, aten.silu, aten.view]
# x_271 => mul_211, sigmoid_64
# x_272 => mean_16
# x_274 => view
triton_per_fused_mean_silu_view_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_view_93', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (144, ), (1, ))
    assert_size_stride(primals_14, (144, ), (1, ))
    assert_size_stride(primals_15, (144, ), (1, ))
    assert_size_stride(primals_16, (144, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (144, ), (1, ))
    assert_size_stride(primals_20, (144, ), (1, ))
    assert_size_stride(primals_21, (144, ), (1, ))
    assert_size_stride(primals_22, (144, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_24, (40, ), (1, ))
    assert_size_stride(primals_25, (240, ), (1, ))
    assert_size_stride(primals_26, (240, ), (1, ))
    assert_size_stride(primals_27, (240, ), (1, ))
    assert_size_stride(primals_28, (240, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_30, (40, ), (1, ))
    assert_size_stride(primals_31, (240, ), (1, ))
    assert_size_stride(primals_32, (240, ), (1, ))
    assert_size_stride(primals_33, (240, ), (1, ))
    assert_size_stride(primals_34, (240, ), (1, ))
    assert_size_stride(primals_35, (80, ), (1, ))
    assert_size_stride(primals_36, (80, ), (1, ))
    assert_size_stride(primals_37, (480, ), (1, ))
    assert_size_stride(primals_38, (480, ), (1, ))
    assert_size_stride(primals_39, (480, ), (1, ))
    assert_size_stride(primals_40, (480, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_42, (80, ), (1, ))
    assert_size_stride(primals_43, (480, ), (1, ))
    assert_size_stride(primals_44, (480, ), (1, ))
    assert_size_stride(primals_45, (480, ), (1, ))
    assert_size_stride(primals_46, (480, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_48, (80, ), (1, ))
    assert_size_stride(primals_49, (480, ), (1, ))
    assert_size_stride(primals_50, (480, ), (1, ))
    assert_size_stride(primals_51, (480, ), (1, ))
    assert_size_stride(primals_52, (480, ), (1, ))
    assert_size_stride(primals_53, (112, ), (1, ))
    assert_size_stride(primals_54, (112, ), (1, ))
    assert_size_stride(primals_55, (672, ), (1, ))
    assert_size_stride(primals_56, (672, ), (1, ))
    assert_size_stride(primals_57, (672, ), (1, ))
    assert_size_stride(primals_58, (672, ), (1, ))
    assert_size_stride(primals_59, (112, ), (1, ))
    assert_size_stride(primals_60, (112, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_62, (672, ), (1, ))
    assert_size_stride(primals_63, (672, ), (1, ))
    assert_size_stride(primals_64, (672, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_66, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_68, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_70, (672, ), (1, ))
    assert_size_stride(primals_71, (192, ), (1, ))
    assert_size_stride(primals_72, (192, ), (1, ))
    assert_size_stride(primals_73, (1152, ), (1, ))
    assert_size_stride(primals_74, (1152, ), (1, ))
    assert_size_stride(primals_75, (1152, ), (1, ))
    assert_size_stride(primals_76, (1152, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_78, (192, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_82, (1152, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_84, (192, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_86, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_88, (1152, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_92, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_94, (1152, ), (1, ))
    assert_size_stride(primals_95, (320, ), (1, ))
    assert_size_stride(primals_96, (320, ), (1, ))
    assert_size_stride(primals_97, (1280, ), (1, ))
    assert_size_stride(primals_98, (1280, ), (1, ))
    assert_size_stride(primals_99, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_100, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_102, (8, ), (1, ))
    assert_size_stride(primals_103, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_104, (32, ), (1, ))
    assert_size_stride(primals_105, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_106, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_107, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_111, (96, ), (1, ))
    assert_size_stride(primals_112, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_113, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_114, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_115, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_116, (6, ), (1, ))
    assert_size_stride(primals_117, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_118, (144, ), (1, ))
    assert_size_stride(primals_119, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_120, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_121, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_123, (6, ), (1, ))
    assert_size_stride(primals_124, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_125, (144, ), (1, ))
    assert_size_stride(primals_126, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_127, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_128, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_129, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_130, (10, ), (1, ))
    assert_size_stride(primals_131, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_132, (240, ), (1, ))
    assert_size_stride(primals_133, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_134, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_135, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_137, (10, ), (1, ))
    assert_size_stride(primals_138, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_139, (240, ), (1, ))
    assert_size_stride(primals_140, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_141, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_142, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_144, (20, ), (1, ))
    assert_size_stride(primals_145, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_146, (480, ), (1, ))
    assert_size_stride(primals_147, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_148, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_149, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_150, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_151, (20, ), (1, ))
    assert_size_stride(primals_152, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_153, (480, ), (1, ))
    assert_size_stride(primals_154, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_155, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_156, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_157, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_158, (20, ), (1, ))
    assert_size_stride(primals_159, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_160, (480, ), (1, ))
    assert_size_stride(primals_161, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_163, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_164, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_165, (28, ), (1, ))
    assert_size_stride(primals_166, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_167, (672, ), (1, ))
    assert_size_stride(primals_168, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_169, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_170, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_171, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_172, (28, ), (1, ))
    assert_size_stride(primals_173, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_174, (672, ), (1, ))
    assert_size_stride(primals_175, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_176, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_177, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_178, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_179, (28, ), (1, ))
    assert_size_stride(primals_180, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_181, (672, ), (1, ))
    assert_size_stride(primals_182, (192, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_183, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_184, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_186, (48, ), (1, ))
    assert_size_stride(primals_187, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_188, (1152, ), (1, ))
    assert_size_stride(primals_189, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_190, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_191, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_192, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_193, (48, ), (1, ))
    assert_size_stride(primals_194, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_195, (1152, ), (1, ))
    assert_size_stride(primals_196, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_197, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_198, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_199, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_200, (48, ), (1, ))
    assert_size_stride(primals_201, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_202, (1152, ), (1, ))
    assert_size_stride(primals_203, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_204, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_205, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (48, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_207, (48, ), (1, ))
    assert_size_stride(primals_208, (1152, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_209, (1152, ), (1, ))
    assert_size_stride(primals_210, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_211, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_212, (1000, 1280), (1280, 1))
    assert_size_stride(primals_213, (1000, ), (1, ))
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, ), (1, ))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (16, ), (1, ))
    assert_size_stride(primals_220, (96, ), (1, ))
    assert_size_stride(primals_221, (96, ), (1, ))
    assert_size_stride(primals_222, (96, ), (1, ))
    assert_size_stride(primals_223, (96, ), (1, ))
    assert_size_stride(primals_224, (24, ), (1, ))
    assert_size_stride(primals_225, (24, ), (1, ))
    assert_size_stride(primals_226, (144, ), (1, ))
    assert_size_stride(primals_227, (144, ), (1, ))
    assert_size_stride(primals_228, (144, ), (1, ))
    assert_size_stride(primals_229, (144, ), (1, ))
    assert_size_stride(primals_230, (24, ), (1, ))
    assert_size_stride(primals_231, (24, ), (1, ))
    assert_size_stride(primals_232, (144, ), (1, ))
    assert_size_stride(primals_233, (144, ), (1, ))
    assert_size_stride(primals_234, (144, ), (1, ))
    assert_size_stride(primals_235, (144, ), (1, ))
    assert_size_stride(primals_236, (40, ), (1, ))
    assert_size_stride(primals_237, (40, ), (1, ))
    assert_size_stride(primals_238, (240, ), (1, ))
    assert_size_stride(primals_239, (240, ), (1, ))
    assert_size_stride(primals_240, (240, ), (1, ))
    assert_size_stride(primals_241, (240, ), (1, ))
    assert_size_stride(primals_242, (40, ), (1, ))
    assert_size_stride(primals_243, (40, ), (1, ))
    assert_size_stride(primals_244, (240, ), (1, ))
    assert_size_stride(primals_245, (240, ), (1, ))
    assert_size_stride(primals_246, (240, ), (1, ))
    assert_size_stride(primals_247, (240, ), (1, ))
    assert_size_stride(primals_248, (80, ), (1, ))
    assert_size_stride(primals_249, (80, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_251, (480, ), (1, ))
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_255, (80, ), (1, ))
    assert_size_stride(primals_256, (480, ), (1, ))
    assert_size_stride(primals_257, (480, ), (1, ))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_259, (480, ), (1, ))
    assert_size_stride(primals_260, (80, ), (1, ))
    assert_size_stride(primals_261, (80, ), (1, ))
    assert_size_stride(primals_262, (480, ), (1, ))
    assert_size_stride(primals_263, (480, ), (1, ))
    assert_size_stride(primals_264, (480, ), (1, ))
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (112, ), (1, ))
    assert_size_stride(primals_267, (112, ), (1, ))
    assert_size_stride(primals_268, (672, ), (1, ))
    assert_size_stride(primals_269, (672, ), (1, ))
    assert_size_stride(primals_270, (672, ), (1, ))
    assert_size_stride(primals_271, (672, ), (1, ))
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_273, (112, ), (1, ))
    assert_size_stride(primals_274, (672, ), (1, ))
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_276, (672, ), (1, ))
    assert_size_stride(primals_277, (672, ), (1, ))
    assert_size_stride(primals_278, (112, ), (1, ))
    assert_size_stride(primals_279, (112, ), (1, ))
    assert_size_stride(primals_280, (672, ), (1, ))
    assert_size_stride(primals_281, (672, ), (1, ))
    assert_size_stride(primals_282, (672, ), (1, ))
    assert_size_stride(primals_283, (672, ), (1, ))
    assert_size_stride(primals_284, (192, ), (1, ))
    assert_size_stride(primals_285, (192, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_287, (1152, ), (1, ))
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_290, (192, ), (1, ))
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (1152, ), (1, ))
    assert_size_stride(primals_293, (1152, ), (1, ))
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_296, (192, ), (1, ))
    assert_size_stride(primals_297, (192, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_299, (1152, ), (1, ))
    assert_size_stride(primals_300, (1152, ), (1, ))
    assert_size_stride(primals_301, (1152, ), (1, ))
    assert_size_stride(primals_302, (192, ), (1, ))
    assert_size_stride(primals_303, (192, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_305, (1152, ), (1, ))
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_308, (320, ), (1, ))
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (1280, ), (1, ))
    assert_size_stride(primals_311, (1280, ), (1, ))
    assert_size_stride(primals_312, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_99, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_99
        buf1 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_312, buf1, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_312
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf3 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf2, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf5 = reinterpret_tensor(buf2, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf2  # reuse
        buf323 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_3.run(buf3, primals_214, primals_215, primals_1, primals_2, buf5, buf323, 1605632, grid=grid(1605632), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf6, (4, 32, 112, 112), (401408, 12544, 112, 1))
        buf7 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_2.run(buf6, buf7, 128, 12544, grid=grid(128, 12544), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (4, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf6  # reuse
        # Source Nodes: [x_6], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf7, primals_216, primals_217, primals_3, primals_4, buf8, 1605632, grid=grid(1605632), stream=stream0)
        del primals_4
        buf9 = empty_strided((4, 32, 1, 1, 98), (3136, 1, 12544, 12544, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_5.run(buf8, buf9, 12544, 128, grid=grid(12544), stream=stream0)
        buf10 = empty_strided((4, 32, 1, 1), (32, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf11 = reinterpret_tensor(buf10, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf10  # reuse
        # Source Nodes: [x_9, x_se], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_6.run(buf11, buf9, 128, 98, grid=grid(128), stream=stream0)
        del buf9
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 8, 1, 1), (8, 1, 1, 1))
        buf13 = reinterpret_tensor(buf12, (4, 8, 1, 1), (8, 1, 8, 8), 0); del buf12  # reuse
        buf14 = empty_strided((4, 8, 1, 1), (8, 1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_7.run(buf13, primals_102, buf14, 32, grid=grid(32), stream=stream0)
        del primals_102
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 32, 1, 1), (32, 1, 1, 1))
        buf16 = reinterpret_tensor(buf15, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf15  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf16, primals_104, 128, grid=grid(128), stream=stream0)
        del primals_104
        buf17 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate, x_10, x_9], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_9.run(buf8, buf16, buf17, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 16, 112, 112), (200704, 12544, 112, 1))
        buf19 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(buf18, buf19, 64, 12544, grid=grid(64, 12544), stream=stream0)
        buf20 = reinterpret_tensor(buf18, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf18  # reuse
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf19, primals_218, primals_219, primals_5, primals_6, buf20, 802816, grid=grid(802816), stream=stream0)
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 96, 112, 112), (1204224, 12544, 112, 1))
        buf22 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf21, buf22, 384, 12544, grid=grid(384, 12544), stream=stream0)
        buf24 = reinterpret_tensor(buf21, (4, 96, 112, 112), (1204224, 1, 10752, 96), 0); del buf21  # reuse
        buf322 = empty_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_13.run(buf22, primals_220, primals_221, primals_7, primals_8, buf24, buf322, 4816896, grid=grid(4816896), stream=stream0)
        del primals_8
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_107, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf25, (4, 96, 56, 56), (301056, 3136, 56, 1))
        buf26 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf25, buf26, 384, 3136, grid=grid(384, 3136), stream=stream0)
        buf27 = reinterpret_tensor(buf25, (4, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf25  # reuse
        # Source Nodes: [x_22], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf26, primals_222, primals_223, primals_9, primals_10, buf27, 1204224, grid=grid(1204224), stream=stream0)
        del primals_10
        buf28 = empty_strided((4, 96, 1, 1, 25), (2400, 1, 9600, 9600, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_16.run(buf27, buf28, 9600, 126, grid=grid(9600), stream=stream0)
        buf29 = empty_strided((4, 96, 1, 1), (96, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf30 = reinterpret_tensor(buf29, (4, 96, 1, 1), (96, 1, 96, 96), 0); del buf29  # reuse
        # Source Nodes: [x_25, x_se_4], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_17.run(buf30, buf28, 384, 25, grid=grid(384), stream=stream0)
        del buf28
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 4, 1, 1), (4, 1, 1, 1))
        buf32 = reinterpret_tensor(buf31, (4, 4, 1, 1), (4, 1, 4, 4), 0); del buf31  # reuse
        buf33 = empty_strided((4, 4, 1, 1), (4, 1, 4, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_18.run(buf32, primals_109, buf33, 16, grid=grid(16), stream=stream0)
        del primals_109
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 96, 1, 1), (96, 1, 1, 1))
        buf35 = reinterpret_tensor(buf34, (4, 96, 1, 1), (96, 1, 96, 96), 0); del buf34  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_19.run(buf35, primals_111, 384, grid=grid(384), stream=stream0)
        del primals_111
        buf36 = empty_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate, x_25, x_26], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_20.run(buf27, buf35, buf36, 1204224, grid=grid(1204224), stream=stream0)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf38 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf37, buf38, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf39 = reinterpret_tensor(buf37, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf37  # reuse
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf38, primals_224, primals_225, primals_11, primals_12, buf39, 301056, grid=grid(301056), stream=stream0)
        del primals_12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf41 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf40, buf41, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf43 = reinterpret_tensor(buf40, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf40  # reuse
        buf321 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_24.run(buf41, primals_226, primals_227, primals_13, primals_14, buf43, buf321, 1806336, grid=grid(1806336), stream=stream0)
        del primals_14
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf44, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf45 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf44, buf45, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf44  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf45, primals_228, primals_229, primals_15, primals_16, buf46, 1806336, grid=grid(1806336), stream=stream0)
        del primals_16
        buf47 = empty_strided((4, 144, 1, 1, 25), (3600, 1, 14400, 14400, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_26.run(buf46, buf47, 14400, 126, grid=grid(14400), stream=stream0)
        buf48 = empty_strided((4, 144, 1, 1), (144, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf49 = reinterpret_tensor(buf48, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf48  # reuse
        # Source Nodes: [x_41, x_se_8], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_27.run(buf49, buf47, 576, 25, grid=grid(576), stream=stream0)
        del buf47
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 6, 1, 1), (6, 1, 1, 1))
        buf51 = reinterpret_tensor(buf50, (4, 6, 1, 1), (6, 1, 6, 6), 0); del buf50  # reuse
        buf52 = empty_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf51, primals_116, buf52, 24, grid=grid(24), stream=stream0)
        del primals_116
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 144, 1, 1), (144, 1, 1, 1))
        buf54 = reinterpret_tensor(buf53, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf53  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf54, primals_118, 576, grid=grid(576), stream=stream0)
        del primals_118
        buf55 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate, x_41, x_42], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_30.run(buf46, buf54, buf55, 1806336, grid=grid(1806336), stream=stream0)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 24, 56, 56), (75264, 3136, 56, 1))
        buf57 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf56, buf57, 96, 3136, grid=grid(96, 3136), stream=stream0)
        buf58 = reinterpret_tensor(buf56, (4, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf56  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_31.run(buf57, primals_230, primals_231, primals_17, primals_18, buf39, buf58, 301056, grid=grid(301056), stream=stream0)
        del primals_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 144, 56, 56), (451584, 3136, 56, 1))
        buf60 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf59, buf60, 576, 3136, grid=grid(576, 3136), stream=stream0)
        buf62 = reinterpret_tensor(buf59, (4, 144, 56, 56), (451584, 1, 8064, 144), 0); del buf59  # reuse
        buf320 = empty_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_24.run(buf60, primals_232, primals_233, primals_19, primals_20, buf62, buf320, 1806336, grid=grid(1806336), stream=stream0)
        del primals_20
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_121, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf63, (4, 144, 28, 28), (112896, 784, 28, 1))
        buf64 = empty_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_32.run(buf63, buf64, 576, 784, grid=grid(576, 784), stream=stream0)
        buf65 = reinterpret_tensor(buf63, (4, 144, 28, 28), (112896, 1, 4032, 144), 0); del buf63  # reuse
        # Source Nodes: [x_55], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf64, primals_234, primals_235, primals_21, primals_22, buf65, 451584, grid=grid(451584), stream=stream0)
        del primals_22
        buf66 = empty_strided((4, 144, 1, 1, 7), (1008, 1, 4032, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_34.run(buf65, buf66, 4032, 112, grid=grid(4032), stream=stream0)
        buf67 = empty_strided((4, 144, 1, 1), (144, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf68 = reinterpret_tensor(buf67, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf67  # reuse
        # Source Nodes: [x_58, x_se_12], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_35.run(buf68, buf66, 576, 7, grid=grid(576), stream=stream0)
        del buf66
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 6, 1, 1), (6, 1, 1, 1))
        buf70 = reinterpret_tensor(buf69, (4, 6, 1, 1), (6, 1, 6, 6), 0); del buf69  # reuse
        buf71 = empty_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf70, primals_123, buf71, 24, grid=grid(24), stream=stream0)
        del primals_123
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 144, 1, 1), (144, 1, 1, 1))
        buf73 = reinterpret_tensor(buf72, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf72  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29.run(buf73, primals_125, 576, grid=grid(576), stream=stream0)
        del primals_125
        buf74 = empty_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_58, x_59], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_36.run(buf65, buf73, buf74, 451584, grid=grid(451584), stream=stream0)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf76 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf75, buf76, 160, 784, grid=grid(160, 784), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf75  # reuse
        # Source Nodes: [x_61], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf76, primals_236, primals_237, primals_23, primals_24, buf77, 125440, grid=grid(125440), stream=stream0)
        del primals_24
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 240, 28, 28), (188160, 784, 28, 1))
        buf79 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf78, buf79, 960, 784, grid=grid(960, 784), stream=stream0)
        buf81 = reinterpret_tensor(buf78, (4, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf78  # reuse
        buf319 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_40.run(buf79, primals_238, primals_239, primals_25, primals_26, buf81, buf319, 752640, grid=grid(752640), stream=stream0)
        del primals_26
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_128, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf82, (4, 240, 28, 28), (188160, 784, 28, 1))
        buf83 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf82, buf83, 960, 784, grid=grid(960, 784), stream=stream0)
        buf84 = reinterpret_tensor(buf82, (4, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf82  # reuse
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf83, primals_240, primals_241, primals_27, primals_28, buf84, 752640, grid=grid(752640), stream=stream0)
        del primals_28
        buf85 = empty_strided((4, 240, 1, 1, 7), (1680, 1, 6720, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_42.run(buf84, buf85, 6720, 112, grid=grid(6720), stream=stream0)
        buf86 = empty_strided((4, 240, 1, 1), (240, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf87 = reinterpret_tensor(buf86, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf86  # reuse
        # Source Nodes: [x_74, x_se_16], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_43.run(buf87, buf85, 960, 7, grid=grid(960), stream=stream0)
        del buf85
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 10, 1, 1), (10, 1, 1, 1))
        buf89 = reinterpret_tensor(buf88, (4, 10, 1, 1), (10, 1, 10, 10), 0); del buf88  # reuse
        buf90 = empty_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_44.run(buf89, primals_130, buf90, 40, grid=grid(40), stream=stream0)
        del primals_130
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 240, 1, 1), (240, 1, 1, 1))
        buf92 = reinterpret_tensor(buf91, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf91  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf92, primals_132, 960, grid=grid(960), stream=stream0)
        del primals_132
        buf93 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_74, x_75], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_46.run(buf84, buf92, buf93, 752640, grid=grid(752640), stream=stream0)
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 40, 28, 28), (31360, 784, 28, 1))
        buf95 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_76], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf94, buf95, 160, 784, grid=grid(160, 784), stream=stream0)
        buf96 = reinterpret_tensor(buf94, (4, 40, 28, 28), (31360, 1, 1120, 40), 0); del buf94  # reuse
        # Source Nodes: [shortcut_5, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf95, primals_242, primals_243, primals_29, primals_30, buf77, buf96, 125440, grid=grid(125440), stream=stream0)
        del primals_30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 240, 28, 28), (188160, 784, 28, 1))
        buf98 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_39.run(buf97, buf98, 960, 784, grid=grid(960, 784), stream=stream0)
        buf100 = reinterpret_tensor(buf97, (4, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf97  # reuse
        buf318 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_40.run(buf98, primals_244, primals_245, primals_31, primals_32, buf100, buf318, 752640, grid=grid(752640), stream=stream0)
        del primals_32
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_135, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf101, (4, 240, 14, 14), (47040, 196, 14, 1))
        buf102 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48.run(buf101, buf102, 960, 196, grid=grid(960, 196), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (4, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf101  # reuse
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_49.run(buf102, primals_246, primals_247, primals_33, primals_34, buf103, 188160, grid=grid(188160), stream=stream0)
        del primals_34
        buf104 = empty_strided((4, 240, 1, 1, 2), (480, 1, 1920, 1920, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_50.run(buf103, buf104, 1920, 98, grid=grid(1920), stream=stream0)
        buf105 = empty_strided((4, 240, 1, 1), (240, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf106 = reinterpret_tensor(buf105, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf105  # reuse
        # Source Nodes: [x_91, x_se_20], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_51.run(buf106, buf104, 960, 2, grid=grid(960), stream=stream0)
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 10, 1, 1), (10, 1, 1, 1))
        buf108 = reinterpret_tensor(buf107, (4, 10, 1, 1), (10, 1, 10, 10), 0); del buf107  # reuse
        buf109 = empty_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_44.run(buf108, primals_137, buf109, 40, grid=grid(40), stream=stream0)
        del primals_137
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 240, 1, 1), (240, 1, 1, 1))
        buf111 = reinterpret_tensor(buf110, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf110  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf111, primals_139, 960, grid=grid(960), stream=stream0)
        del primals_139
        buf112 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_91, x_92], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_52.run(buf103, buf111, buf112, 188160, grid=grid(188160), stream=stream0)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf114 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf113, buf114, 320, 196, grid=grid(320, 196), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf113  # reuse
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_54.run(buf114, primals_248, primals_249, primals_35, primals_36, buf115, 62720, grid=grid(62720), stream=stream0)
        del primals_36
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf117 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf116, buf117, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf119 = reinterpret_tensor(buf116, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf116  # reuse
        buf317 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_56.run(buf117, primals_250, primals_251, primals_37, primals_38, buf119, buf317, 376320, grid=grid(376320), stream=stream0)
        del primals_38
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf120, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf121 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf120, buf121, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf122 = reinterpret_tensor(buf120, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf120  # reuse
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_57.run(buf121, primals_252, primals_253, primals_39, primals_40, buf122, 376320, grid=grid(376320), stream=stream0)
        del primals_40
        buf123 = empty_strided((4, 480, 1, 1, 2), (960, 1, 3840, 3840, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_58.run(buf122, buf123, 3840, 98, grid=grid(3840), stream=stream0)
        buf124 = reinterpret_tensor(buf104, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf104  # reuse
        buf125 = reinterpret_tensor(buf124, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf124  # reuse
        # Source Nodes: [x_107, x_se_24], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_59.run(buf125, buf123, 1920, 2, grid=grid(1920), stream=stream0)
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 20, 1, 1), (20, 1, 1, 1))
        buf127 = reinterpret_tensor(buf126, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf126  # reuse
        buf128 = empty_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_60.run(buf127, primals_144, buf128, 80, grid=grid(80), stream=stream0)
        del primals_144
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 480, 1, 1), (480, 1, 1, 1))
        buf130 = reinterpret_tensor(buf129, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf129  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf130, primals_146, 1920, grid=grid(1920), stream=stream0)
        del primals_146
        buf131 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_107, x_108], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_62.run(buf122, buf130, buf131, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf133 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf132, buf133, 320, 196, grid=grid(320, 196), stream=stream0)
        buf134 = reinterpret_tensor(buf132, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf132  # reuse
        # Source Nodes: [shortcut_7, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_63.run(buf133, primals_254, primals_255, primals_41, primals_42, buf115, buf134, 62720, grid=grid(62720), stream=stream0)
        del primals_42
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf136 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf135, buf136, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf138 = reinterpret_tensor(buf135, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf135  # reuse
        buf316 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_56.run(buf136, primals_256, primals_257, primals_43, primals_44, buf138, buf316, 376320, grid=grid(376320), stream=stream0)
        del primals_44
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_149, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf139, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf140 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf139, buf140, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf139  # reuse
        # Source Nodes: [x_121], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_57.run(buf140, primals_258, primals_259, primals_45, primals_46, buf141, 376320, grid=grid(376320), stream=stream0)
        del primals_46
        buf142 = buf123; del buf123  # reuse
        # Source Nodes: [x_124, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_58.run(buf141, buf142, 3840, 98, grid=grid(3840), stream=stream0)
        buf143 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf144 = reinterpret_tensor(buf143, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf143  # reuse
        # Source Nodes: [x_124, x_se_28], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_59.run(buf144, buf142, 1920, 2, grid=grid(1920), stream=stream0)
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 20, 1, 1), (20, 1, 1, 1))
        buf146 = reinterpret_tensor(buf145, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf145  # reuse
        buf147 = empty_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_60.run(buf146, primals_151, buf147, 80, grid=grid(80), stream=stream0)
        del primals_151
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 480, 1, 1), (480, 1, 1, 1))
        buf149 = reinterpret_tensor(buf148, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf148  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf149, primals_153, 1920, grid=grid(1920), stream=stream0)
        del primals_153
        buf150 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_124, x_125], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_62.run(buf141, buf149, buf150, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 80, 14, 14), (15680, 196, 14, 1))
        buf152 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf151, buf152, 320, 196, grid=grid(320, 196), stream=stream0)
        buf153 = reinterpret_tensor(buf151, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf151  # reuse
        # Source Nodes: [shortcut_8, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_63.run(buf152, primals_260, primals_261, primals_47, primals_48, buf134, buf153, 62720, grid=grid(62720), stream=stream0)
        del primals_48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_155, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf155 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf154, buf155, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf157 = reinterpret_tensor(buf154, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf154  # reuse
        buf315 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_56.run(buf155, primals_262, primals_263, primals_49, primals_50, buf157, buf315, 376320, grid=grid(376320), stream=stream0)
        del primals_50
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_156, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf158, (4, 480, 14, 14), (94080, 196, 14, 1))
        buf159 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_55.run(buf158, buf159, 1920, 196, grid=grid(1920, 196), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf158  # reuse
        # Source Nodes: [x_138], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_57.run(buf159, primals_264, primals_265, primals_51, primals_52, buf160, 376320, grid=grid(376320), stream=stream0)
        del primals_52
        buf161 = buf142; del buf142  # reuse
        # Source Nodes: [x_141, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_58.run(buf160, buf161, 3840, 98, grid=grid(3840), stream=stream0)
        buf162 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf163 = reinterpret_tensor(buf162, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf162  # reuse
        # Source Nodes: [x_141, x_se_32], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_59.run(buf163, buf161, 1920, 2, grid=grid(1920), stream=stream0)
        del buf161
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 20, 1, 1), (20, 1, 1, 1))
        buf165 = reinterpret_tensor(buf164, (4, 20, 1, 1), (20, 1, 20, 20), 0); del buf164  # reuse
        buf166 = empty_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_60.run(buf165, primals_158, buf166, 80, grid=grid(80), stream=stream0)
        del primals_158
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 480, 1, 1), (480, 1, 1, 1))
        buf168 = reinterpret_tensor(buf167, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf167  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_61.run(buf168, primals_160, 1920, grid=grid(1920), stream=stream0)
        del primals_160
        buf169 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_141, x_142], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_62.run(buf160, buf168, buf169, 376320, grid=grid(376320), stream=stream0)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 112, 14, 14), (21952, 196, 14, 1))
        buf171 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf170, buf171, 448, 196, grid=grid(448, 196), stream=stream0)
        buf172 = reinterpret_tensor(buf170, (4, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf170  # reuse
        # Source Nodes: [x_144], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_65.run(buf171, primals_266, primals_267, primals_53, primals_54, buf172, 87808, grid=grid(87808), stream=stream0)
        del primals_54
        # Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf174 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf173, buf174, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf176 = reinterpret_tensor(buf173, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf173  # reuse
        buf314 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_67.run(buf174, primals_268, primals_269, primals_55, primals_56, buf176, buf314, 526848, grid=grid(526848), stream=stream0)
        del primals_56
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_163, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf177, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf178 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf177, buf178, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf179 = reinterpret_tensor(buf177, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf177  # reuse
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_68.run(buf178, primals_270, primals_271, primals_57, primals_58, buf179, 526848, grid=grid(526848), stream=stream0)
        del primals_58
        buf180 = empty_strided((4, 672, 1, 1, 2), (1344, 1, 5376, 5376, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_69.run(buf179, buf180, 5376, 98, grid=grid(5376), stream=stream0)
        buf181 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf182 = reinterpret_tensor(buf181, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf181  # reuse
        # Source Nodes: [x_157, x_se_36], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_70.run(buf182, buf180, 2688, 2, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 28, 1, 1), (28, 1, 1, 1))
        buf184 = reinterpret_tensor(buf183, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf183  # reuse
        buf185 = empty_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_71.run(buf184, primals_165, buf185, 112, grid=grid(112), stream=stream0)
        del primals_165
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 672, 1, 1), (672, 1, 1, 1))
        buf187 = reinterpret_tensor(buf186, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf186  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf187, primals_167, 2688, grid=grid(2688), stream=stream0)
        del primals_167
        buf188 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_157, x_158], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_73.run(buf179, buf187, buf188, 526848, grid=grid(526848), stream=stream0)
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 112, 14, 14), (21952, 196, 14, 1))
        buf190 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf189, buf190, 448, 196, grid=grid(448, 196), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (4, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf189  # reuse
        # Source Nodes: [shortcut_10, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_74.run(buf190, primals_272, primals_273, primals_59, primals_60, buf172, buf191, 87808, grid=grid(87808), stream=stream0)
        del primals_60
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf193 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf192, buf193, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf195 = reinterpret_tensor(buf192, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf192  # reuse
        buf313 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_67.run(buf193, primals_274, primals_275, primals_61, primals_62, buf195, buf313, 526848, grid=grid(526848), stream=stream0)
        del primals_62
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_170, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf196, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf197 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf196, buf197, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf196  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_68.run(buf197, primals_276, primals_277, primals_63, primals_64, buf198, 526848, grid=grid(526848), stream=stream0)
        del primals_64
        buf199 = buf180; del buf180  # reuse
        # Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_red_fused_mean_silu_69.run(buf198, buf199, 5376, 98, grid=grid(5376), stream=stream0)
        buf200 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf201 = reinterpret_tensor(buf200, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf200  # reuse
        # Source Nodes: [x_174, x_se_40], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_70.run(buf201, buf199, 2688, 2, grid=grid(2688), stream=stream0)
        del buf199
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 28, 1, 1), (28, 1, 1, 1))
        buf203 = reinterpret_tensor(buf202, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf202  # reuse
        buf204 = empty_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_71.run(buf203, primals_172, buf204, 112, grid=grid(112), stream=stream0)
        del primals_172
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 672, 1, 1), (672, 1, 1, 1))
        buf206 = reinterpret_tensor(buf205, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf205  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf206, primals_174, 2688, grid=grid(2688), stream=stream0)
        del primals_174
        buf207 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_174, x_175], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_73.run(buf198, buf206, buf207, 526848, grid=grid(526848), stream=stream0)
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 112, 14, 14), (21952, 196, 14, 1))
        buf209 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_64.run(buf208, buf209, 448, 196, grid=grid(448, 196), stream=stream0)
        buf210 = reinterpret_tensor(buf208, (4, 112, 14, 14), (21952, 1, 1568, 112), 0); del buf208  # reuse
        # Source Nodes: [shortcut_11, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_74.run(buf209, primals_278, primals_279, primals_65, primals_66, buf191, buf210, 87808, grid=grid(87808), stream=stream0)
        del primals_66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 672, 14, 14), (131712, 196, 14, 1))
        buf212 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_66.run(buf211, buf212, 2688, 196, grid=grid(2688, 196), stream=stream0)
        buf214 = reinterpret_tensor(buf211, (4, 672, 14, 14), (131712, 1, 9408, 672), 0); del buf211  # reuse
        buf312 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_67.run(buf212, primals_280, primals_281, primals_67, primals_68, buf214, buf312, 526848, grid=grid(526848), stream=stream0)
        del primals_68
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_177, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf215, (4, 672, 7, 7), (32928, 49, 7, 1))
        buf216 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_75.run(buf215, buf216, 2688, 49, grid=grid(2688, 49), stream=stream0)
        buf217 = reinterpret_tensor(buf215, (4, 672, 7, 7), (32928, 1, 4704, 672), 0); del buf215  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_76.run(buf216, primals_282, primals_283, primals_69, primals_70, buf217, 131712, grid=grid(131712), stream=stream0)
        del primals_70
        buf218 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf219 = reinterpret_tensor(buf218, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf218  # reuse
        # Source Nodes: [x_191, x_se_44], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_77.run(buf219, buf217, 2688, 49, grid=grid(2688), stream=stream0)
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 28, 1, 1), (28, 1, 1, 1))
        buf221 = reinterpret_tensor(buf220, (4, 28, 1, 1), (28, 1, 28, 28), 0); del buf220  # reuse
        buf222 = empty_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_71.run(buf221, primals_179, buf222, 112, grid=grid(112), stream=stream0)
        del primals_179
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 672, 1, 1), (672, 1, 1, 1))
        buf224 = reinterpret_tensor(buf223, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf223  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_72.run(buf224, primals_181, 2688, grid=grid(2688), stream=stream0)
        del primals_181
        buf225 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_191, x_192], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_78.run(buf217, buf224, buf225, 131712, grid=grid(131712), stream=stream0)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf227 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf226, buf227, 768, 49, grid=grid(768, 49), stream=stream0)
        buf228 = reinterpret_tensor(buf226, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf226  # reuse
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_80.run(buf227, primals_284, primals_285, primals_71, primals_72, buf228, 37632, grid=grid(37632), stream=stream0)
        del primals_72
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf230 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf229, buf230, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf232 = reinterpret_tensor(buf229, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf229  # reuse
        buf311 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82.run(buf230, primals_286, primals_287, primals_73, primals_74, buf232, buf311, 225792, grid=grid(225792), stream=stream0)
        del primals_74
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_184, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf233, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf234 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf233, buf234, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf235 = reinterpret_tensor(buf233, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf233  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_83.run(buf234, primals_288, primals_289, primals_75, primals_76, buf235, 225792, grid=grid(225792), stream=stream0)
        del primals_76
        buf236 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cuda', dtype=torch.float32)
        buf237 = reinterpret_tensor(buf236, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf236  # reuse
        # Source Nodes: [x_207, x_se_48], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_84.run(buf237, buf235, 4608, 49, grid=grid(4608), stream=stream0)
        # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 48, 1, 1), (48, 1, 1, 1))
        buf239 = reinterpret_tensor(buf238, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf238  # reuse
        buf240 = empty_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_49, x_se_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_85.run(buf239, primals_186, buf240, 192, grid=grid(192), stream=stream0)
        del primals_186
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 1152, 1, 1), (1152, 1, 1, 1))
        buf242 = reinterpret_tensor(buf241, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf241  # reuse
        # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_86.run(buf242, primals_188, 4608, grid=grid(4608), stream=stream0)
        del primals_188
        buf243 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_207, x_208], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_87.run(buf235, buf242, buf243, 225792, grid=grid(225792), stream=stream0)
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf245 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_209], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf244, buf245, 768, 49, grid=grid(768, 49), stream=stream0)
        buf246 = reinterpret_tensor(buf244, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf244  # reuse
        # Source Nodes: [shortcut_13, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_88.run(buf245, primals_290, primals_291, primals_77, primals_78, buf228, buf246, 37632, grid=grid(37632), stream=stream0)
        del primals_78
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf248 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf247, buf248, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf250 = reinterpret_tensor(buf247, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf247  # reuse
        buf310 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82.run(buf248, primals_292, primals_293, primals_79, primals_80, buf250, buf310, 225792, grid=grid(225792), stream=stream0)
        del primals_80
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_191, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf251, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf252 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf251, buf252, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf253 = reinterpret_tensor(buf251, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf251  # reuse
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_83.run(buf252, primals_294, primals_295, primals_81, primals_82, buf253, 225792, grid=grid(225792), stream=stream0)
        del primals_82
        buf254 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cuda', dtype=torch.float32)
        buf255 = reinterpret_tensor(buf254, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf254  # reuse
        # Source Nodes: [x_224, x_se_52], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_84.run(buf255, buf253, 4608, 49, grid=grid(4608), stream=stream0)
        # Source Nodes: [x_se_53], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 48, 1, 1), (48, 1, 1, 1))
        buf257 = reinterpret_tensor(buf256, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf256  # reuse
        buf258 = empty_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_53, x_se_54], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_85.run(buf257, primals_193, buf258, 192, grid=grid(192), stream=stream0)
        del primals_193
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 1152, 1, 1), (1152, 1, 1, 1))
        buf260 = reinterpret_tensor(buf259, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf259  # reuse
        # Source Nodes: [x_se_55], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_86.run(buf260, primals_195, 4608, grid=grid(4608), stream=stream0)
        del primals_195
        buf261 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_224, x_225], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_87.run(buf253, buf260, buf261, 225792, grid=grid(225792), stream=stream0)
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf263 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf262, buf263, 768, 49, grid=grid(768, 49), stream=stream0)
        buf264 = reinterpret_tensor(buf262, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf262  # reuse
        # Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_88.run(buf263, primals_296, primals_297, primals_83, primals_84, buf246, buf264, 37632, grid=grid(37632), stream=stream0)
        del primals_84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf266 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf265, buf266, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf268 = reinterpret_tensor(buf265, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf265  # reuse
        buf309 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82.run(buf266, primals_298, primals_299, primals_85, primals_86, buf268, buf309, 225792, grid=grid(225792), stream=stream0)
        del primals_86
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_198, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf269, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf270 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf269, buf270, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf269  # reuse
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_83.run(buf270, primals_300, primals_301, primals_87, primals_88, buf271, 225792, grid=grid(225792), stream=stream0)
        del primals_88
        buf272 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cuda', dtype=torch.float32)
        buf273 = reinterpret_tensor(buf272, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf272  # reuse
        # Source Nodes: [x_241, x_se_56], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_84.run(buf273, buf271, 4608, 49, grid=grid(4608), stream=stream0)
        # Source Nodes: [x_se_57], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (4, 48, 1, 1), (48, 1, 1, 1))
        buf275 = reinterpret_tensor(buf274, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf274  # reuse
        buf276 = empty_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_57, x_se_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_85.run(buf275, primals_200, buf276, 192, grid=grid(192), stream=stream0)
        del primals_200
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (4, 1152, 1, 1), (1152, 1, 1, 1))
        buf278 = reinterpret_tensor(buf277, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf277  # reuse
        # Source Nodes: [x_se_59], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_86.run(buf278, primals_202, 4608, grid=grid(4608), stream=stream0)
        del primals_202
        buf279 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_241, x_242], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_87.run(buf271, buf278, buf279, 225792, grid=grid(225792), stream=stream0)
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 192, 7, 7), (9408, 49, 7, 1))
        buf281 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_79.run(buf280, buf281, 768, 49, grid=grid(768, 49), stream=stream0)
        buf282 = reinterpret_tensor(buf280, (4, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf280  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_88.run(buf281, primals_302, primals_303, primals_89, primals_90, buf264, buf282, 37632, grid=grid(37632), stream=stream0)
        del primals_90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf284 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf283, buf284, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf286 = reinterpret_tensor(buf283, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf283  # reuse
        buf308 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_silu_sub_82.run(buf284, primals_304, primals_305, primals_91, primals_92, buf286, buf308, 225792, grid=grid(225792), stream=stream0)
        del primals_92
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf287, (4, 1152, 7, 7), (56448, 49, 7, 1))
        buf288 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_81.run(buf287, buf288, 4608, 49, grid=grid(4608, 49), stream=stream0)
        buf289 = reinterpret_tensor(buf287, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf287  # reuse
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_83.run(buf288, primals_306, primals_307, primals_93, primals_94, buf289, 225792, grid=grid(225792), stream=stream0)
        del primals_94
        buf290 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cuda', dtype=torch.float32)
        buf291 = reinterpret_tensor(buf290, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf290  # reuse
        # Source Nodes: [x_258, x_se_60], Original ATen: [aten.mean, aten.silu]
        triton_per_fused_mean_silu_84.run(buf291, buf289, 4608, 49, grid=grid(4608), stream=stream0)
        # Source Nodes: [x_se_61], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 48, 1, 1), (48, 1, 1, 1))
        buf293 = reinterpret_tensor(buf292, (4, 48, 1, 1), (48, 1, 48, 48), 0); del buf292  # reuse
        buf294 = empty_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_se_61, x_se_62], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_85.run(buf293, primals_207, buf294, 192, grid=grid(192), stream=stream0)
        del primals_207
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 1152, 1, 1), (1152, 1, 1, 1))
        buf296 = reinterpret_tensor(buf295, (4, 1152, 1, 1), (1152, 1, 1152, 1152), 0); del buf295  # reuse
        # Source Nodes: [x_se_63], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_86.run(buf296, primals_209, 4608, grid=grid(4608), stream=stream0)
        del primals_209
        buf297 = empty_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate, x_258, x_259], Original ATen: [aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_mul_sigmoid_silu_87.run(buf289, buf296, buf297, 225792, grid=grid(225792), stream=stream0)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 320, 7, 7), (15680, 49, 7, 1))
        buf299 = empty_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_89.run(buf298, buf299, 1280, 49, grid=grid(1280, 49), stream=stream0)
        buf300 = reinterpret_tensor(buf298, (4, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf298  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_90.run(buf299, primals_308, primals_309, primals_95, primals_96, buf300, 62720, grid=grid(62720), stream=stream0)
        del primals_96
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 1280, 7, 7), (62720, 49, 7, 1))
        buf302 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_91.run(buf301, buf302, 5120, 49, grid=grid(5120, 49), stream=stream0)
        buf303 = reinterpret_tensor(buf301, (4, 1280, 7, 7), (62720, 1, 8960, 1280), 0); del buf301  # reuse
        buf307 = empty_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_no_training_add_fill_mul_sigmoid_sub_92.run(buf302, primals_310, primals_311, primals_97, primals_98, buf303, buf307, 250880, grid=grid(250880), stream=stream0)
        del primals_98
        buf304 = empty_strided((4, 1280, 1, 1), (1280, 1, 5120, 5120), device='cuda', dtype=torch.float32)
        buf305 = reinterpret_tensor(buf304, (4, 1280), (1280, 1), 0); del buf304  # reuse
        # Source Nodes: [x_271, x_272, x_274], Original ATen: [aten.mean, aten.silu, aten.view]
        triton_per_fused_mean_silu_view_93.run(buf305, buf303, 5120, 49, grid=grid(5120), stream=stream0)
        del buf303
        buf306 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, buf305, reinterpret_tensor(primals_212, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf306)
        del primals_213
        return (buf306, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, buf0, primals_100, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_119, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_154, primals_155, primals_156, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_169, primals_170, primals_171, primals_173, primals_175, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, buf1, buf3, buf5, buf7, buf8, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf24, buf26, buf27, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf41, buf43, buf45, buf46, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf62, buf64, buf65, buf68, buf70, buf71, buf73, buf74, buf76, buf77, buf79, buf81, buf83, buf84, buf87, buf89, buf90, buf92, buf93, buf95, buf96, buf98, buf100, buf102, buf103, buf106, buf108, buf109, buf111, buf112, buf114, buf115, buf117, buf119, buf121, buf122, buf125, buf127, buf128, buf130, buf131, buf133, buf134, buf136, buf138, buf140, buf141, buf144, buf146, buf147, buf149, buf150, buf152, buf153, buf155, buf157, buf159, buf160, buf163, buf165, buf166, buf168, buf169, buf171, buf172, buf174, buf176, buf178, buf179, buf182, buf184, buf185, buf187, buf188, buf190, buf191, buf193, buf195, buf197, buf198, buf201, buf203, buf204, buf206, buf207, buf209, buf210, buf212, buf214, buf216, buf217, buf219, buf221, buf222, buf224, buf225, buf227, buf228, buf230, buf232, buf234, buf235, buf237, buf239, buf240, buf242, buf243, buf245, buf246, buf248, buf250, buf252, buf253, buf255, buf257, buf258, buf260, buf261, buf263, buf264, buf266, buf268, buf270, buf271, buf273, buf275, buf276, buf278, buf279, buf281, buf282, buf284, buf286, buf288, buf289, buf291, buf293, buf294, buf296, buf297, buf299, buf300, buf302, buf305, reinterpret_tensor(primals_212, (1000, 1280), (1280, 1), 0), buf307, buf308, buf309, buf310, buf311, buf312, buf313, buf314, buf315, buf316, buf317, buf318, buf319, buf320, buf321, buf322, buf323, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((192, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((48, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1152, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_efficientnet', benchmark_compiled_module)
