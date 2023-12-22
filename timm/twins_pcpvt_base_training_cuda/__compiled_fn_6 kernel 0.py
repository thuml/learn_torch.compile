
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


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtcbt4g57hrviadj7jcv3toxxh4xcnjikjw4nylbesjf4s5onol.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/43/c43tkg2l5npzjta5izpuxxs32lenaw35ext2fakkj22yqmuk4fpb.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 49.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 512.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszszajb2n5dc6zte2xgrxwfty2ek53kbinwa5kxszpdhrfzu5bj.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*(r2 // 49)) + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdvr3cf2dgo36cscmkajlhwvgxzvyi32ep5hkjc6qvsdr5rlybk.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_per_fused_div_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpam6lmzv35vy6s4ccizlvwxxtoyjzpyoqzzyd3uq3z36cqnxmr.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 49)
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnnq7jctvgyzufotgllryo2rvutldx34xsvc6xm4qqk3tppbtka.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5fyrsdupkfyjoaxxgfvpevh42jxs6i6buvbrzlwgpa6jdt2jdz.py
# Source Nodes: [x_467], Original ATen: [aten.gelu, aten.gelu_backward]
# x_467 => add_256, erf_27, mul_252
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casol26iuba3s2avpya6i4w5w2duebpxdu3eprxq63eo5uy2bldz.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (200704*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qogtm527csp2gwqgzz35722rxgyyvumvkhxugdvfih2t6zgtk7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3gayv6fg4yf5i5v7gujy6xyftcbvjfw4gopyuz2khwm64v76ug.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 392
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2cc7p5msksl62ejiznksnfdpt7ktawphk54qtz6txtj2koicx2.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuve7vl4b4jvy4ze2bn55mxyi37iuzgkgzbje7rcsmnvlzaoia6r.py
# Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]

triton_poi_fused__scaled_dot_product_efficient_attention_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmshutoyobfcxrdxwizvoyqtkxsvzk7obzotw2soqo6ofmtjqfu.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 25088)
    x6 = xindex
    x0 = xindex % 512
    x3 = (xindex // 200704)
    x7 = (xindex // 512) % 392
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-200704) + x6), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (512*x3) + (1024*x7)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfu5p7cb5uj6ypc7v7gv3twl2p6s2guhhzr5wgts5u2tw2nso46.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (100352*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce65jz66hggl7w3mbqm2o7lwsnfr6yh4x5xg37p34fvlekjsdbgq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu56x4vs5d5onklqcelhe5cue73pdgwnojnqqh6qdrip5sih6bea.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ev/cevbvxjgnt5bk6feiou3ggsa7sb64o64gm4defftg5zr3x3pmwrq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 392
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = 512.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdobozney7igtxqiheuwa36rwwkooqs3od355p2tn75fku4ff3c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp8 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62s4qj2yviqoyq3lvmhaqe3pqr5jmtqmahurhunnbmgvyc6ujil.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (25088*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg25ym6tsgitytuz2uuj6p5ft2bmwgasqmyqokmk727yawz2mhmt.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((49*x1) + (25088*(y0 // 49)) + (y0 % 49)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdg5bp6ocwqlnhs7j26isn6toppdkb3iulbtfi6zuw7nukgw3imd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 392
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
    x2 = xindex % 49
    x3 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (49*r1) + (25088*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnt3tc7wq7aaw7jwnlbr23o6glf7mtepffecmjjnzl4gw2dk4qa.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, rnumel):
    xnumel = 392
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = 512.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp25 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp37 = tmp25 * tmp17
    tmp38 = tmp37 - tmp29
    tmp39 = tmp30 * tmp35
    tmp40 = tmp38 - tmp39
    tmp41 = tmp36 * tmp40
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24d4htc2djilq2dzrpxhh4xlxkzxopyznn4x4bohqtwwogxjae5.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (62720*(y0 // 196)) + (y0 % 196)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (320*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dqqgr5qiu5hnxegqcn2knhiaprvls6sy4bfixbyy6rijhmzqss.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*r2) + (38720*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wk/cwkmecw7dsopocx66upq2krldtpl3fd5qklgrppcvftcnrw73ynq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f7/cf72zxixhc4e4fjxcxucmydb7zjugp32ixpmqmr4r33ov54jlpue.py
# Source Nodes: [x_419], Original ATen: [aten.gelu, aten.gelu_backward]
# x_419 => add_232, erf_24, mul_229
triton_poi_fused_gelu_gelu_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxstnk36nlzb7qslwzrgiudgwiuvztkrevwcloqnbp2e5o4ft6o2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16640
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1280)
    x0 = xindex % 1280
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1280*r2) + (154880*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/ciljfnbi35o252j63vtwlxc2ltdn4wyidnnmqwzjl7gq5zqh6prp.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2b7gxepdlljhkpuet4avgg7ys7i5f7tstinlv6w5m2xa7qpnujd.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (196*r1) + (62720*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 320.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (320*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ij/cijyf4kmbim7uat5khw4nm3h23noedhrmksy6k5vdecco43dcp7r.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp12 = tl.where(tmp2, tmp3, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6ywm7m2flcbokvam4uwvx5as4sjjt75ozm5sfns3n3vx3akjyg.py
# Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]

triton_poi_fused__scaled_dot_product_efficient_attention_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 5
    y1 = (yindex // 5)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (5*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7xkgc5pkrypwkwdnsg3t555urcnrionerh5pwvgwg2fofi22pz.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 15680)
    x6 = xindex
    x0 = xindex % 320
    x3 = (xindex // 125440)
    x7 = (xindex // 320) % 392
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-125440) + x6), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (320*x3) + (640*x7)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5f/c5f32ad5dc4tjelfuqtp77j7rykkgpma2evix6xmd4jmxpefxg5r.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 640
    x1 = (xindex // 640)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (62720*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccv4huxggu2au2yz6yfnrjpbpxg3c7cjauzbr742rbwiqbdcz2e2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxpccyysw7ynzbpgbud6dagrxa2aa2n7eqrjxjdki4nrqp7jzvgo.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 320.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlhsn3hjayy3mxby44w6xx7yhboo43oeibqgtnwghmmcmfgsgak.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfccqdm56intpbmoa557fxavh5hhbb2gllmrh4pphba7qekwhxde.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cusxzgbaoeyluvaexkpfgfcybki3zhb637ocequ3ou7u2tx3fv4j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 320
    x1 = (xindex // 320)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (31360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cufnbnmtj3hcbpocudcovv5pt3uehuvkzjbtvesue6eraj3cvkd5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x4 = (xindex // 3)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x0) + (320*x4)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r3 + (107*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfluhrjtyor4eexbgi64sgnjhybtm5e5gutxwcxdsigqo2h443cl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc6c46n52ymihz3a2fhtxzm5feruo22x2d6bubw2g73jncswoy5l.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = 320.0
    tmp14 = tmp4 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp5 * tmp10
    tmp18 = tmp16 - tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp11 + tmp19
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (320*x3)), tmp20, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52hv2to4ci2rvtp6bxl3fe6ld7zj5kk5occwepqmx2ayv74r5ya.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (62720*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6u7bwvdjsiimzdoce42xc2dce6frstypbiiwxiavtjqn2rbgyr.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (62720*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (320*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipopwiphh7bjqwwfxyf2t7472yykddrdfeu4fbgmkurb7qxjshn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxgcofcsdq2y6jl5afabj245zbkfocphuwh336cvzzu2h63l3xu.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 320.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eo/ceokxjmqtu4t7st6kj23e7sm43dsed6har5hj4wsc27arjw2jbia.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = 320.0
    tmp14 = tmp4 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp5 * tmp10
    tmp18 = tmp16 - tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp11 + tmp19
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2uzgjnipaj3m2ry66v6hx3siq2pejkxfj3ym7dn2uonoxuckot.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (320*x2) + (62720*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewfrzqsqveymxljnorfptktkuyfiti7aiiyjob5xmeii2xr2lxk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (62720*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sucgyxorkn43aw5rw5lo6f4jirylf3hl4hi4vhx5xbzws264ux.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (320*x2) + (62720*y1)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwk5ii4y4yd7d5sqjbefdz5mex4r64p2fwvi5znkwjgm4pkrrva.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (196*r1) + (62720*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x2 + (196*r1) + (62720*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = tmp13 + tmp14
    tmp17 = 320.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp6
    tmp20 = tmp7 * tmp12
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (320*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ux/cuxrzhbu7eilkdgfxqaobigkyqohvgekycbswmloov6rva2fh6na.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
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
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr7 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = 320.0
    tmp14 = tmp4 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp5 * tmp10
    tmp18 = tmp16 - tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp11 + tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tmp22 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp22 * tmp13
    tmp35 = tmp34 - tmp26
    tmp36 = tmp27 * tmp32
    tmp37 = tmp35 - tmp36
    tmp38 = tmp33 * tmp37
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp38, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajw5sgkniznrvzmptpqvmxzi25jtmntxqozgfuoiznhmlc4gpso.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vh/cvh2vapyp5vn5lcnm73yzcugj6uw4rnch6ajrogve52j2vmemrgq.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (100352*(y0 // 784)) + (y0 % 784)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czribxnbswrl65zerbrjh4xnhhitumuprejdaocgpsgojjjni3sw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrzehtljtnuctm3d2kfqet56dsojx7gn7cgf4lquegvpdjhdks4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqi2muqpu3c7isnpmzx5ar72excmvcaaxxp56yvuhxu77md6hob.py
# Source Nodes: [x_122], Original ATen: [aten.gelu, aten.gelu_backward]
# x_122 => add_67, erf_6, mul_65
triton_poi_fused_gelu_gelu_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gotyzwildshbilwcl7i6tivat2m3kenhijllc7eglo34hxzc4f.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxha5myxocqvvggcvkqibyee3ulky763zv52xr4nnfm4e7crzn6r.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fb4vcwcgkg25zftcl7ethzy22j7oic2a2wm2xc7iyjczreti6w.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (784*r1) + (100352*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6ghf2alwkqfskyckkxr6iwudlptiym3ngcnjbaqv4n7jibwsrx.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqpbhnd64w6aul355uwniwashtxr42tyiwg2dlbkc7eqqetpu5g.py
# Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]

triton_poi_fused__scaled_dot_product_efficient_attention_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = (yindex // 2)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (50176*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xz4j62d4w3p3h4ex6djlav3w2oea6bwbo3tz3lay22mfsupzb7.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 6272)
    x6 = xindex
    x0 = xindex % 128
    x3 = (xindex // 50176)
    x7 = (xindex // 128) % 392
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-50176) + x6), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (128*x3) + (256*x7)), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coe7witcoqaf3xibtzpds6cvfx5sppkff3rqqqq2yzyilfjswm3j.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/celcwbg3xxze5bgx7o5r7jepbnwank6lenqgmdkmugj5pt7objyv.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzitr7jfjsfea2jxepxsihvsdue32weyxe3742yinqmmvtydvd6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 128.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2ohhs7qehuateyyq56w7dxz2lwtbmnvkksi5356ejl4gzonx3a.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccx7nfxafaxmgsfurdcskbezrj7siuv4o2jpsswinmyd7673f2s7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq56p4wxyzgh3dnbfh6vofe2wlwrka6vj65u2jdd4p3fhvunlpzk.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjjtxlyy5auqm5kpqhjv67vjcshqoxaywkj4segziqk6hl6hhvw.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_red_fused__unsafe_view_add_clone_native_layer_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_native_layer_norm_backward_68', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = 128.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp6
        tmp24 = tmp23 * tmp11
        tmp25 = tmp22 - tmp24
        tmp26 = tmp14 * tmp25
        tmp27 = tmp13 + tmp26
        tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp27, rmask & xmask)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yugexdmnhaod63iwrgo7g6u72muafahmsdlmk5f772fbrlsmuu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (100352*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chfflnudwhdma4k5cher5w7p44vf4jpppuwr6rl3t2pa6yallqbq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (100352*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacrppemluacnk3lah6ctzi2ifgyy2eue5pf2i64cqbc7ztr35an.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crndnpm5jy2lyfnhwly3opt6fva7h3vby457hnc4miollnoliejk.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3hv3bj3dp3vsqvwshj4sptmsfil5i6o2vdaspvxtkea3275bsm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_73', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = 128.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp6
        tmp24 = tmp23 * tmp11
        tmp25 = tmp22 - tmp24
        tmp26 = tmp14 * tmp25
        tmp27 = tmp13 + tmp26
        tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjop5ir2cm7uuhhfgk2e3vdsqp5fw47bjhsaibclwgu3d4qlseg.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6plqsrjcyinmlx5ewcefvyyjrdapyfzwzxqoiy2qg6xm4gcibbm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4gfokkjk4k5ouxkq2efkeghghpfjwm2ciifkxt2z2fvvtptb6f.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23lrn2db6alxukmpcuhwhaf5n2ufsmd22we7osavn5724gxtcbx.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (784*r1) + (100352*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x2 + (784*r1) + (100352*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp13 + tmp14
    tmp17 = 128.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp6
    tmp20 = tmp7 * tmp12
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4cuprtayulqwwx5mtzmhqhhd27mmnesoendyy2ciwln2buho5j.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_78', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = 128.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp6
        tmp24 = tmp23 * tmp11
        tmp25 = tmp22 - tmp24
        tmp26 = tmp14 * tmp25
        tmp27 = tmp13 + tmp26
        tmp29 = tmp27 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp27, rmask & xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    _tmp39 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp33 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tl.load(in_ptr6 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 * tmp36
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(rmask & xmask, tmp40, _tmp39)
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tmp41 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp42 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr6 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp42 * tmp43
        tmp45 = 128.0
        tmp46 = tmp44 * tmp45
        tmp47 = tmp46 - tmp31
        tmp49 = tmp48 * tmp39
        tmp50 = tmp47 - tmp49
        tmp51 = tmp41 * tmp50
        tl.store(out_ptr4 + (r2 + (128*x3)), tmp51, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhcl4n33pft2l7ijdsi76lrv4so3k33w5bs6pgitv7iut6zve6t.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncmtnpcjzvj37md5eq5wx46vjc7jgq5wu3lneiobzkkkilo2lzn.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_80 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((3136*x1) + (200704*(y0 // 3136)) + (y0 % 3136)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (64*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffwgih7omhewvlmzktjrpqxug6sq6zpr34uwcizrd4ta2ql5sqd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxcexnqmbomk2esd2z6z4hh4cb5otw2ptlt2v5onjzjz4phh5wew.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckoccavmwh4ibzu2pygerygaksqbof534ehqk2yudd2s67lgjkzz.py
# Source Nodes: [x_49], Original ATen: [aten.gelu, aten.gelu_backward]
# x_49 => add_28, erf_2, mul_27
triton_poi_fused_gelu_gelu_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l3/cl3wjlju75chcegw5ic3o7i6mk27jhnvexyrwkyewliy3aqzprzs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cab5ppvvgwiy4sfft7u7bpuqzmqih5lsvym5j575bpkt7vocrslw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceurij2sscnt36mprswd5fxshj3plvl76dbxdfvub5ponyv35tym.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_86', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 3136
    x3 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (3136*r1) + (200704*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 64.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxyxfpclfjsvtp27qdev26c6kczovww7qhuox65xx52jnw56aqn.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnbskrckhersxbv66jyak3wyrf42ioi2u7erqxh5niyrhrlyz2v.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 3136)
    x6 = xindex
    x0 = xindex % 64
    x3 = (xindex // 25088)
    x7 = (xindex // 64) % 392
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-25088) + x6), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (64*x3) + (128*x7)), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cav73ak35jgwgwrtmgsefm4gt7codpuqlub5yidyiu6tcei3aqt3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlmp6x63y675ncj764kpnzo6swifafxwiywpq3tlrfq2gzskixn.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = 64.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmluwcayasbkeocmmcrfsek7mgbcx6nfkfyk2dx7hmlewkd7yuh3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_91 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwhrnsn5qxv4kofu5i5ltbork23aof5bq5pehv4kbcrmrkmadty.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfd6h7n6s2wbk7aedd5v2bjyinppsjgr2vdv5bcgnhixahsilrm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (6272*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chcjvjdbuapsz6xsy4r4pj2albvc6nibjp6rscnk53j4kznrunxz.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_94', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp23, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbuh5gtsmr45cr4bfvcz65hjyhoioz3saz6bcesojl6s4ibekue.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2k275wccalqkatd5dy2jdkgfrf2qxuvwzpuvgaxtekguijp5g2l.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7i/c7ii5bdegzybz5gx3bvap5p2nkwqbpy2ldr6mxdlqrklzu66aufg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4p5y7x57xjfht3nl6uzjrn7wxq7lq7j5yx55ed4xm54dfjhiqi.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_98 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_98', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 64.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqww2f5fipojelm5kqqqtf6w4qzdjmpr77czunpszbohw7o7zo4r.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_99', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqgajjqlh6chglavk52ctspehx54nzu54v7mm5qoxzwfe7tmdrt.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xun5gdb2qxwco7o745ktc6zjvh3kqfrqdmzcdcz25nufhky4mz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (200704*(r2 // 3136)) + (401408*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cys6soc6xz2ffysves2o6ogu33f5yl2dsmn7dmw7qcf5aanetbbl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gcty5j74wsv3jvu6rwhzcj3v5ufgye2dxyq5x7vyjvoq5iw643.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumhunmhjywaaaun3a2mfsnlejzcwavxvlgg72nsvfem4pswbk6q.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_104 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 3136
    x3 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (3136*r1) + (200704*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x2 + (3136*r1) + (200704*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp13 + tmp14
    tmp17 = 64.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp6
    tmp20 = tmp7 * tmp12
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gz/cgzaqo2fef6asuwj7veqtqi7i4mfm4yf2v7fzf2eerh2ioslcpqb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_105', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tmp25 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp37 = tmp25 * tmp17
    tmp38 = tmp37 - tmp29
    tmp39 = tmp30 * tmp35
    tmp40 = tmp38 - tmp39
    tmp41 = tmp36 * tmp40
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp23, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (64*x3)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2pagjrra4mb2bwlwo56w2uyh4wuitsgac6wqx3dvgicu53tdfq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_9, primals_11, primals_17, primals_23, primals_25, primals_29, primals_31, primals_37, primals_43, primals_47, primals_49, primals_55, primals_61, primals_63, primals_65, primals_69, primals_71, primals_77, primals_83, primals_85, primals_89, primals_91, primals_97, primals_103, primals_107, primals_109, primals_115, primals_121, primals_125, primals_127, primals_133, primals_139, primals_141, primals_143, primals_147, primals_149, primals_155, primals_161, primals_163, primals_167, primals_169, primals_175, primals_181, primals_185, primals_187, primals_193, primals_199, primals_203, primals_205, primals_211, primals_217, primals_221, primals_223, primals_229, primals_235, primals_239, primals_241, primals_247, primals_253, primals_257, primals_259, primals_265, primals_271, primals_275, primals_277, primals_283, primals_289, primals_293, primals_295, primals_301, primals_307, primals_311, primals_313, primals_319, primals_325, primals_329, primals_331, primals_337, primals_343, primals_347, primals_349, primals_355, primals_361, primals_365, primals_367, primals_373, primals_379, primals_383, primals_385, primals_391, primals_397, primals_401, primals_403, primals_409, primals_415, primals_419, primals_421, primals_427, primals_433, primals_437, primals_439, primals_445, primals_451, primals_455, primals_457, primals_463, primals_469, primals_471, primals_473, primals_481, primals_487, primals_489, primals_497, primals_503, primals_511, primals_517, primals_521, mul, mul_2, view_1, permute_2, view_4, mul_4, view_6, getitem_6, getitem_7, getitem_9, getitem_10, getitem_11, view_10, mul_6, view_12, addmm_3, view_14, view_16, mul_11, view_19, permute_15, view_22, mul_13, view_24, getitem_18, getitem_19, getitem_21, getitem_22, getitem_23, view_28, mul_15, view_30, addmm_8, view_32, mul_20, view_34, permute_25, view_37, mul_22, view_39, getitem_30, getitem_31, getitem_33, getitem_34, getitem_35, view_43, mul_24, view_45, addmm_13, view_47, permute_34, mul_29, mul_31, view_51, permute_37, view_54, mul_33, view_56, getitem_44, getitem_45, getitem_47, getitem_48, getitem_49, view_60, mul_35, view_62, addmm_18, view_64, view_66, mul_40, view_69, permute_50, view_72, mul_42, view_74, getitem_56, getitem_57, getitem_59, getitem_60, getitem_61, view_78, mul_44, view_80, addmm_23, view_82, mul_49, view_84, permute_60, view_87, mul_51, view_89, getitem_68, getitem_69, getitem_71, getitem_72, getitem_73, view_93, mul_53, view_95, addmm_28, view_97, mul_58, view_99, permute_70, view_102, mul_60, view_104, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_108, mul_62, view_110, addmm_33, view_112, permute_79, mul_67, mul_69, view_116, permute_82, view_119, mul_71, view_121, getitem_94, getitem_95, getitem_97, getitem_98, getitem_99, view_125, mul_73, view_127, addmm_38, view_129, view_131, mul_78, view_134, permute_95, view_137, mul_80, view_139, getitem_106, getitem_107, getitem_109, getitem_110, getitem_111, view_143, mul_82, view_145, addmm_43, view_147, mul_87, view_149, permute_105, view_152, mul_89, view_154, getitem_118, getitem_119, getitem_121, getitem_122, getitem_123, view_158, mul_91, view_160, addmm_48, view_162, mul_96, view_164, permute_115, view_167, mul_98, view_169, getitem_130, getitem_131, getitem_133, getitem_134, getitem_135, view_173, mul_100, view_175, addmm_53, view_177, mul_105, view_179, permute_125, view_182, mul_107, view_184, getitem_142, getitem_143, getitem_145, getitem_146, getitem_147, view_188, mul_109, view_190, addmm_58, view_192, mul_114, view_194, permute_135, view_197, mul_116, view_199, getitem_154, getitem_155, getitem_157, getitem_158, getitem_159, view_203, mul_118, view_205, addmm_63, view_207, mul_123, view_209, permute_145, view_212, mul_125, view_214, getitem_166, getitem_167, getitem_169, getitem_170, getitem_171, view_218, mul_127, view_220, addmm_68, view_222, mul_132, view_224, permute_155, view_227, mul_134, view_229, getitem_178, getitem_179, getitem_181, getitem_182, getitem_183, view_233, mul_136, view_235, addmm_73, view_237, mul_141, view_239, permute_165, view_242, mul_143, view_244, getitem_190, getitem_191, getitem_193, getitem_194, getitem_195, view_248, mul_145, view_250, addmm_78, view_252, mul_150, view_254, permute_175, view_257, mul_152, view_259, getitem_202, getitem_203, getitem_205, getitem_206, getitem_207, view_263, mul_154, view_265, addmm_83, view_267, mul_159, view_269, permute_185, view_272, mul_161, view_274, getitem_214, getitem_215, getitem_217, getitem_218, getitem_219, view_278, mul_163, view_280, addmm_88, view_282, mul_168, view_284, permute_195, view_287, mul_170, view_289, getitem_226, getitem_227, getitem_229, getitem_230, getitem_231, view_293, mul_172, view_295, addmm_93, view_297, mul_177, view_299, permute_205, view_302, mul_179, view_304, getitem_238, getitem_239, getitem_241, getitem_242, getitem_243, view_308, mul_181, view_310, addmm_98, view_312, mul_186, view_314, permute_215, view_317, mul_188, view_319, getitem_250, getitem_251, getitem_253, getitem_254, getitem_255, view_323, mul_190, view_325, addmm_103, view_327, mul_195, view_329, permute_225, view_332, mul_197, view_334, getitem_262, getitem_263, getitem_265, getitem_266, getitem_267, view_338, mul_199, view_340, addmm_108, view_342, mul_204, view_344, permute_235, view_347, mul_206, view_349, getitem_274, getitem_275, getitem_277, getitem_278, getitem_279, view_353, mul_208, view_355, addmm_113, view_357, mul_213, view_359, permute_245, view_362, mul_215, view_364, getitem_286, getitem_287, getitem_289, getitem_290, getitem_291, view_368, mul_217, view_370, addmm_118, view_372, mul_222, view_374, permute_255, view_377, mul_224, view_379, getitem_298, getitem_299, getitem_301, getitem_302, getitem_303, view_383, mul_226, view_385, addmm_123, view_387, permute_264, mul_231, mul_233, view_391, permute_267, getitem_310, getitem_311, getitem_313, getitem_314, getitem_315, view_398, mul_235, view_400, addmm_128, view_402, view_404, mul_240, view_407, permute_278, getitem_320, getitem_321, getitem_323, getitem_324, getitem_325, view_414, mul_242, view_416, addmm_133, view_418, mul_247, view_420, permute_286, getitem_330, getitem_331, getitem_333, getitem_334, getitem_335, view_427, mul_249, view_429, addmm_138, view_431, mul_254, clone_166, permute_294, div_1, permute_298, permute_302, div_2, permute_306, alias_28, permute_312, permute_317, div_3, permute_321, permute_325, div_4, permute_329, alias_29, permute_335, permute_340, div_5, permute_346, permute_350, div_6, permute_354, alias_30, permute_360, permute_365, div_7, div_8, permute_371, permute_375, div_9, permute_379, alias_31, permute_385, div_10, permute_392, div_11, permute_396, permute_400, div_12, permute_404, alias_32, permute_410, div_13, permute_417, div_14, permute_421, permute_425, div_15, permute_429, alias_33, permute_435, div_16, permute_442, div_17, permute_446, permute_450, div_18, permute_454, alias_34, permute_460, div_19, permute_467, div_20, permute_471, permute_475, div_21, permute_479, alias_35, permute_485, div_22, permute_492, div_23, permute_496, permute_500, div_24, permute_504, alias_36, permute_510, div_25, permute_517, div_26, permute_521, permute_525, div_27, permute_529, alias_37, permute_535, div_28, permute_542, div_29, permute_546, permute_550, div_30, permute_554, alias_38, permute_560, div_31, permute_567, div_32, permute_571, permute_575, div_33, permute_579, alias_39, permute_585, div_34, permute_592, div_35, permute_596, permute_600, div_36, permute_604, alias_40, permute_610, div_37, permute_617, div_38, permute_621, permute_625, div_39, permute_629, alias_41, permute_635, div_40, permute_642, div_41, permute_646, permute_650, div_42, permute_654, alias_42, permute_660, div_43, permute_667, div_44, permute_671, permute_675, div_45, permute_679, alias_43, permute_685, div_46, permute_692, div_47, permute_696, permute_700, div_48, permute_704, alias_44, permute_710, div_49, permute_717, div_50, permute_721, permute_725, div_51, permute_729, alias_45, permute_735, div_52, permute_742, div_53, permute_746, permute_750, div_54, permute_754, alias_46, permute_760, div_55, permute_767, div_56, permute_771, permute_775, div_57, permute_779, alias_47, permute_785, div_58, permute_792, div_59, permute_798, permute_802, div_60, permute_806, alias_48, permute_812, div_61, permute_819, div_62, div_63, permute_825, permute_829, div_64, permute_833, alias_49, permute_839, div_65, permute_846, div_66, permute_850, permute_854, div_67, permute_858, alias_50, permute_864, div_68, permute_871, div_69, permute_875, permute_879, div_70, permute_883, alias_51, permute_889, div_71, permute_896, div_72, permute_902, permute_906, div_73, permute_910, alias_52, permute_916, div_74, permute_923, div_75, div_76, permute_929, permute_933, div_77, permute_937, alias_53, permute_943, div_78, permute_950, div_79, permute_954, permute_958, div_80, permute_962, alias_54, permute_968, div_81, permute_975, div_82, permute_981, permute_985, div_83, permute_989, alias_55, permute_995, div_84, permute_1002, div_85, div_86, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 8, 8), (4096, 1, 512, 64))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_23, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_29, (64, 64, 8, 8), (4096, 1, 512, 64))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_47, (64, 64, 8, 8), (4096, 1, 512, 64))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_61, (128, 64, 2, 2), (256, 1, 128, 64))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_69, (128, 128, 4, 4), (2048, 1, 512, 128))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_83, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_89, (128, 128, 4, 4), (2048, 1, 512, 128))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_107, (128, 128, 4, 4), (2048, 1, 512, 128))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_125, (128, 128, 4, 4), (2048, 1, 512, 128))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_139, (320, 128, 2, 2), (512, 1, 256, 128))
    assert_size_stride(primals_141, (320, ), (1, ))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_147, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_149, (320, ), (1, ))
    assert_size_stride(primals_155, (320, ), (1, ))
    assert_size_stride(primals_161, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_163, (320, ), (1, ))
    assert_size_stride(primals_167, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_169, (320, ), (1, ))
    assert_size_stride(primals_175, (320, ), (1, ))
    assert_size_stride(primals_181, (320, ), (1, ))
    assert_size_stride(primals_185, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_187, (320, ), (1, ))
    assert_size_stride(primals_193, (320, ), (1, ))
    assert_size_stride(primals_199, (320, ), (1, ))
    assert_size_stride(primals_203, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_205, (320, ), (1, ))
    assert_size_stride(primals_211, (320, ), (1, ))
    assert_size_stride(primals_217, (320, ), (1, ))
    assert_size_stride(primals_221, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_223, (320, ), (1, ))
    assert_size_stride(primals_229, (320, ), (1, ))
    assert_size_stride(primals_235, (320, ), (1, ))
    assert_size_stride(primals_239, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_241, (320, ), (1, ))
    assert_size_stride(primals_247, (320, ), (1, ))
    assert_size_stride(primals_253, (320, ), (1, ))
    assert_size_stride(primals_257, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_259, (320, ), (1, ))
    assert_size_stride(primals_265, (320, ), (1, ))
    assert_size_stride(primals_271, (320, ), (1, ))
    assert_size_stride(primals_275, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_277, (320, ), (1, ))
    assert_size_stride(primals_283, (320, ), (1, ))
    assert_size_stride(primals_289, (320, ), (1, ))
    assert_size_stride(primals_293, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_295, (320, ), (1, ))
    assert_size_stride(primals_301, (320, ), (1, ))
    assert_size_stride(primals_307, (320, ), (1, ))
    assert_size_stride(primals_311, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_313, (320, ), (1, ))
    assert_size_stride(primals_319, (320, ), (1, ))
    assert_size_stride(primals_325, (320, ), (1, ))
    assert_size_stride(primals_329, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_331, (320, ), (1, ))
    assert_size_stride(primals_337, (320, ), (1, ))
    assert_size_stride(primals_343, (320, ), (1, ))
    assert_size_stride(primals_347, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_349, (320, ), (1, ))
    assert_size_stride(primals_355, (320, ), (1, ))
    assert_size_stride(primals_361, (320, ), (1, ))
    assert_size_stride(primals_365, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_367, (320, ), (1, ))
    assert_size_stride(primals_373, (320, ), (1, ))
    assert_size_stride(primals_379, (320, ), (1, ))
    assert_size_stride(primals_383, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_385, (320, ), (1, ))
    assert_size_stride(primals_391, (320, ), (1, ))
    assert_size_stride(primals_397, (320, ), (1, ))
    assert_size_stride(primals_401, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_403, (320, ), (1, ))
    assert_size_stride(primals_409, (320, ), (1, ))
    assert_size_stride(primals_415, (320, ), (1, ))
    assert_size_stride(primals_419, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_421, (320, ), (1, ))
    assert_size_stride(primals_427, (320, ), (1, ))
    assert_size_stride(primals_433, (320, ), (1, ))
    assert_size_stride(primals_437, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_439, (320, ), (1, ))
    assert_size_stride(primals_445, (320, ), (1, ))
    assert_size_stride(primals_451, (320, ), (1, ))
    assert_size_stride(primals_455, (320, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_457, (320, ), (1, ))
    assert_size_stride(primals_463, (320, ), (1, ))
    assert_size_stride(primals_469, (512, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_471, (512, ), (1, ))
    assert_size_stride(primals_473, (512, ), (1, ))
    assert_size_stride(primals_481, (512, ), (1, ))
    assert_size_stride(primals_487, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_497, (512, ), (1, ))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_511, (512, ), (1, ))
    assert_size_stride(primals_517, (512, ), (1, ))
    assert_size_stride(primals_521, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(mul_2, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_1, (25088, 64), (64, 1))
    assert_size_stride(permute_2, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(view_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_4, (8, 49, 64), (3136, 64, 1))
    assert_size_stride(view_6, (392, 64), (64, 1))
    assert_size_stride(getitem_6, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_7, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_9, (8, 1, 3136), (3136, 3136, 1))
    assert_size_stride(getitem_10, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(view_10, (25088, 64), (64, 1))
    assert_size_stride(mul_6, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_12, (25088, 64), (64, 1))
    assert_size_stride(addmm_3, (25088, 512), (512, 1))
    assert_size_stride(view_14, (25088, 512), (512, 1))
    assert_size_stride(view_16, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_11, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_19, (25088, 64), (64, 1))
    assert_size_stride(permute_15, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(view_22, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_13, (8, 49, 64), (3136, 64, 1))
    assert_size_stride(view_24, (392, 64), (64, 1))
    assert_size_stride(getitem_18, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_19, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_21, (8, 1, 3136), (3136, 3136, 1))
    assert_size_stride(getitem_22, (), ())
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(view_28, (25088, 64), (64, 1))
    assert_size_stride(mul_15, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_30, (25088, 64), (64, 1))
    assert_size_stride(addmm_8, (25088, 512), (512, 1))
    assert_size_stride(view_32, (25088, 512), (512, 1))
    assert_size_stride(mul_20, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_34, (25088, 64), (64, 1))
    assert_size_stride(permute_25, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(view_37, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_22, (8, 49, 64), (3136, 64, 1))
    assert_size_stride(view_39, (392, 64), (64, 1))
    assert_size_stride(getitem_30, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_31, (8, 1, 49, 64), (6272, 0, 128, 1))
    assert_size_stride(getitem_33, (8, 1, 3136), (3136, 3136, 1))
    assert_size_stride(getitem_34, (), ())
    assert_size_stride(getitem_35, (), ())
    assert_size_stride(view_43, (25088, 64), (64, 1))
    assert_size_stride(mul_24, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_45, (25088, 64), (64, 1))
    assert_size_stride(addmm_13, (25088, 512), (512, 1))
    assert_size_stride(view_47, (25088, 512), (512, 1))
    assert_size_stride(permute_34, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_29, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(mul_31, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_51, (6272, 128), (128, 1))
    assert_size_stride(permute_37, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(view_54, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_33, (8, 49, 128), (6272, 128, 1))
    assert_size_stride(view_56, (392, 128), (128, 1))
    assert_size_stride(getitem_44, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_45, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_47, (8, 2, 800), (1600, 800, 1))
    assert_size_stride(getitem_48, (), ())
    assert_size_stride(getitem_49, (), ())
    assert_size_stride(view_60, (6272, 128), (128, 1))
    assert_size_stride(mul_35, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_62, (6272, 128), (128, 1))
    assert_size_stride(addmm_18, (6272, 1024), (1024, 1))
    assert_size_stride(view_64, (6272, 1024), (1024, 1))
    assert_size_stride(view_66, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_40, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_69, (6272, 128), (128, 1))
    assert_size_stride(permute_50, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(view_72, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_42, (8, 49, 128), (6272, 128, 1))
    assert_size_stride(view_74, (392, 128), (128, 1))
    assert_size_stride(getitem_56, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_57, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_59, (8, 2, 800), (1600, 800, 1))
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(getitem_61, (), ())
    assert_size_stride(view_78, (6272, 128), (128, 1))
    assert_size_stride(mul_44, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_80, (6272, 128), (128, 1))
    assert_size_stride(addmm_23, (6272, 1024), (1024, 1))
    assert_size_stride(view_82, (6272, 1024), (1024, 1))
    assert_size_stride(mul_49, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_84, (6272, 128), (128, 1))
    assert_size_stride(permute_60, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(view_87, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_51, (8, 49, 128), (6272, 128, 1))
    assert_size_stride(view_89, (392, 128), (128, 1))
    assert_size_stride(getitem_68, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_69, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_71, (8, 2, 800), (1600, 800, 1))
    assert_size_stride(getitem_72, (), ())
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(view_93, (6272, 128), (128, 1))
    assert_size_stride(mul_53, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_95, (6272, 128), (128, 1))
    assert_size_stride(addmm_28, (6272, 1024), (1024, 1))
    assert_size_stride(view_97, (6272, 1024), (1024, 1))
    assert_size_stride(mul_58, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_99, (6272, 128), (128, 1))
    assert_size_stride(permute_70, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(view_102, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_60, (8, 49, 128), (6272, 128, 1))
    assert_size_stride(view_104, (392, 128), (128, 1))
    assert_size_stride(getitem_80, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_81, (8, 2, 49, 64), (12544, 64, 256, 1))
    assert_size_stride(getitem_83, (8, 2, 800), (1600, 800, 1))
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(view_108, (6272, 128), (128, 1))
    assert_size_stride(mul_62, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_110, (6272, 128), (128, 1))
    assert_size_stride(addmm_33, (6272, 1024), (1024, 1))
    assert_size_stride(view_112, (6272, 1024), (1024, 1))
    assert_size_stride(permute_79, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_67, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(mul_69, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_116, (1568, 320), (320, 1))
    assert_size_stride(permute_82, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_119, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_71, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_121, (392, 320), (320, 1))
    assert_size_stride(getitem_94, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_95, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_97, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_98, (), ())
    assert_size_stride(getitem_99, (), ())
    assert_size_stride(view_125, (1568, 320), (320, 1))
    assert_size_stride(mul_73, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_127, (1568, 320), (320, 1))
    assert_size_stride(addmm_38, (1568, 1280), (1280, 1))
    assert_size_stride(view_129, (1568, 1280), (1280, 1))
    assert_size_stride(view_131, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_78, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_134, (1568, 320), (320, 1))
    assert_size_stride(permute_95, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_137, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_80, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_139, (392, 320), (320, 1))
    assert_size_stride(getitem_106, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_107, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_109, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_110, (), ())
    assert_size_stride(getitem_111, (), ())
    assert_size_stride(view_143, (1568, 320), (320, 1))
    assert_size_stride(mul_82, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_145, (1568, 320), (320, 1))
    assert_size_stride(addmm_43, (1568, 1280), (1280, 1))
    assert_size_stride(view_147, (1568, 1280), (1280, 1))
    assert_size_stride(mul_87, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_149, (1568, 320), (320, 1))
    assert_size_stride(permute_105, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_152, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_89, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_154, (392, 320), (320, 1))
    assert_size_stride(getitem_118, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_119, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_121, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_122, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(view_158, (1568, 320), (320, 1))
    assert_size_stride(mul_91, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_160, (1568, 320), (320, 1))
    assert_size_stride(addmm_48, (1568, 1280), (1280, 1))
    assert_size_stride(view_162, (1568, 1280), (1280, 1))
    assert_size_stride(mul_96, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_164, (1568, 320), (320, 1))
    assert_size_stride(permute_115, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_167, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_98, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_169, (392, 320), (320, 1))
    assert_size_stride(getitem_130, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_131, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_133, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_134, (), ())
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(view_173, (1568, 320), (320, 1))
    assert_size_stride(mul_100, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_175, (1568, 320), (320, 1))
    assert_size_stride(addmm_53, (1568, 1280), (1280, 1))
    assert_size_stride(view_177, (1568, 1280), (1280, 1))
    assert_size_stride(mul_105, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_179, (1568, 320), (320, 1))
    assert_size_stride(permute_125, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_182, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_107, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_184, (392, 320), (320, 1))
    assert_size_stride(getitem_142, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_143, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_145, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_146, (), ())
    assert_size_stride(getitem_147, (), ())
    assert_size_stride(view_188, (1568, 320), (320, 1))
    assert_size_stride(mul_109, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_190, (1568, 320), (320, 1))
    assert_size_stride(addmm_58, (1568, 1280), (1280, 1))
    assert_size_stride(view_192, (1568, 1280), (1280, 1))
    assert_size_stride(mul_114, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_194, (1568, 320), (320, 1))
    assert_size_stride(permute_135, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_197, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_116, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_199, (392, 320), (320, 1))
    assert_size_stride(getitem_154, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_155, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_157, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_158, (), ())
    assert_size_stride(getitem_159, (), ())
    assert_size_stride(view_203, (1568, 320), (320, 1))
    assert_size_stride(mul_118, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_205, (1568, 320), (320, 1))
    assert_size_stride(addmm_63, (1568, 1280), (1280, 1))
    assert_size_stride(view_207, (1568, 1280), (1280, 1))
    assert_size_stride(mul_123, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_209, (1568, 320), (320, 1))
    assert_size_stride(permute_145, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_212, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_125, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_214, (392, 320), (320, 1))
    assert_size_stride(getitem_166, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_167, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_169, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_170, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(view_218, (1568, 320), (320, 1))
    assert_size_stride(mul_127, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_220, (1568, 320), (320, 1))
    assert_size_stride(addmm_68, (1568, 1280), (1280, 1))
    assert_size_stride(view_222, (1568, 1280), (1280, 1))
    assert_size_stride(mul_132, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_224, (1568, 320), (320, 1))
    assert_size_stride(permute_155, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_227, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_134, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_229, (392, 320), (320, 1))
    assert_size_stride(getitem_178, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_179, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_181, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_182, (), ())
    assert_size_stride(getitem_183, (), ())
    assert_size_stride(view_233, (1568, 320), (320, 1))
    assert_size_stride(mul_136, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_235, (1568, 320), (320, 1))
    assert_size_stride(addmm_73, (1568, 1280), (1280, 1))
    assert_size_stride(view_237, (1568, 1280), (1280, 1))
    assert_size_stride(mul_141, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_239, (1568, 320), (320, 1))
    assert_size_stride(permute_165, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_242, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_143, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_244, (392, 320), (320, 1))
    assert_size_stride(getitem_190, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_191, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_193, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_194, (), ())
    assert_size_stride(getitem_195, (), ())
    assert_size_stride(view_248, (1568, 320), (320, 1))
    assert_size_stride(mul_145, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_250, (1568, 320), (320, 1))
    assert_size_stride(addmm_78, (1568, 1280), (1280, 1))
    assert_size_stride(view_252, (1568, 1280), (1280, 1))
    assert_size_stride(mul_150, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_254, (1568, 320), (320, 1))
    assert_size_stride(permute_175, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_257, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_152, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_259, (392, 320), (320, 1))
    assert_size_stride(getitem_202, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_203, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_205, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_206, (), ())
    assert_size_stride(getitem_207, (), ())
    assert_size_stride(view_263, (1568, 320), (320, 1))
    assert_size_stride(mul_154, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_265, (1568, 320), (320, 1))
    assert_size_stride(addmm_83, (1568, 1280), (1280, 1))
    assert_size_stride(view_267, (1568, 1280), (1280, 1))
    assert_size_stride(mul_159, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_269, (1568, 320), (320, 1))
    assert_size_stride(permute_185, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_272, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_161, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_274, (392, 320), (320, 1))
    assert_size_stride(getitem_214, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_215, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_217, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_218, (), ())
    assert_size_stride(getitem_219, (), ())
    assert_size_stride(view_278, (1568, 320), (320, 1))
    assert_size_stride(mul_163, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_280, (1568, 320), (320, 1))
    assert_size_stride(addmm_88, (1568, 1280), (1280, 1))
    assert_size_stride(view_282, (1568, 1280), (1280, 1))
    assert_size_stride(mul_168, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_284, (1568, 320), (320, 1))
    assert_size_stride(permute_195, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_287, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_170, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_289, (392, 320), (320, 1))
    assert_size_stride(getitem_226, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_227, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_229, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_230, (), ())
    assert_size_stride(getitem_231, (), ())
    assert_size_stride(view_293, (1568, 320), (320, 1))
    assert_size_stride(mul_172, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_295, (1568, 320), (320, 1))
    assert_size_stride(addmm_93, (1568, 1280), (1280, 1))
    assert_size_stride(view_297, (1568, 1280), (1280, 1))
    assert_size_stride(mul_177, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_299, (1568, 320), (320, 1))
    assert_size_stride(permute_205, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_302, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_179, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_304, (392, 320), (320, 1))
    assert_size_stride(getitem_238, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_239, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_241, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_242, (), ())
    assert_size_stride(getitem_243, (), ())
    assert_size_stride(view_308, (1568, 320), (320, 1))
    assert_size_stride(mul_181, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_310, (1568, 320), (320, 1))
    assert_size_stride(addmm_98, (1568, 1280), (1280, 1))
    assert_size_stride(view_312, (1568, 1280), (1280, 1))
    assert_size_stride(mul_186, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_314, (1568, 320), (320, 1))
    assert_size_stride(permute_215, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_317, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_188, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_319, (392, 320), (320, 1))
    assert_size_stride(getitem_250, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_251, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_253, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_254, (), ())
    assert_size_stride(getitem_255, (), ())
    assert_size_stride(view_323, (1568, 320), (320, 1))
    assert_size_stride(mul_190, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_325, (1568, 320), (320, 1))
    assert_size_stride(addmm_103, (1568, 1280), (1280, 1))
    assert_size_stride(view_327, (1568, 1280), (1280, 1))
    assert_size_stride(mul_195, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_329, (1568, 320), (320, 1))
    assert_size_stride(permute_225, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_332, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_197, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_334, (392, 320), (320, 1))
    assert_size_stride(getitem_262, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_263, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_265, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_266, (), ())
    assert_size_stride(getitem_267, (), ())
    assert_size_stride(view_338, (1568, 320), (320, 1))
    assert_size_stride(mul_199, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_340, (1568, 320), (320, 1))
    assert_size_stride(addmm_108, (1568, 1280), (1280, 1))
    assert_size_stride(view_342, (1568, 1280), (1280, 1))
    assert_size_stride(mul_204, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_344, (1568, 320), (320, 1))
    assert_size_stride(permute_235, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_347, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_206, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_349, (392, 320), (320, 1))
    assert_size_stride(getitem_274, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_275, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_277, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_278, (), ())
    assert_size_stride(getitem_279, (), ())
    assert_size_stride(view_353, (1568, 320), (320, 1))
    assert_size_stride(mul_208, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_355, (1568, 320), (320, 1))
    assert_size_stride(addmm_113, (1568, 1280), (1280, 1))
    assert_size_stride(view_357, (1568, 1280), (1280, 1))
    assert_size_stride(mul_213, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_359, (1568, 320), (320, 1))
    assert_size_stride(permute_245, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_362, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_215, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_364, (392, 320), (320, 1))
    assert_size_stride(getitem_286, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_287, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_289, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_290, (), ())
    assert_size_stride(getitem_291, (), ())
    assert_size_stride(view_368, (1568, 320), (320, 1))
    assert_size_stride(mul_217, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_370, (1568, 320), (320, 1))
    assert_size_stride(addmm_118, (1568, 1280), (1280, 1))
    assert_size_stride(view_372, (1568, 1280), (1280, 1))
    assert_size_stride(mul_222, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_374, (1568, 320), (320, 1))
    assert_size_stride(permute_255, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(view_377, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_224, (8, 49, 320), (15680, 320, 1))
    assert_size_stride(view_379, (392, 320), (320, 1))
    assert_size_stride(getitem_298, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_299, (8, 5, 49, 64), (31360, 64, 640, 1))
    assert_size_stride(getitem_301, (8, 5, 224), (1120, 224, 1))
    assert_size_stride(getitem_302, (), ())
    assert_size_stride(getitem_303, (), ())
    assert_size_stride(view_383, (1568, 320), (320, 1))
    assert_size_stride(mul_226, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_385, (1568, 320), (320, 1))
    assert_size_stride(addmm_123, (1568, 1280), (1280, 1))
    assert_size_stride(view_387, (1568, 1280), (1280, 1))
    assert_size_stride(permute_264, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_231, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(mul_233, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_391, (392, 512), (512, 1))
    assert_size_stride(permute_267, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(getitem_310, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_311, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_313, (8, 8, 64), (512, 64, 1))
    assert_size_stride(getitem_314, (), ())
    assert_size_stride(getitem_315, (), ())
    assert_size_stride(view_398, (392, 512), (512, 1))
    assert_size_stride(mul_235, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_400, (392, 512), (512, 1))
    assert_size_stride(addmm_128, (392, 2048), (2048, 1))
    assert_size_stride(view_402, (392, 2048), (2048, 1))
    assert_size_stride(view_404, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(mul_240, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_407, (392, 512), (512, 1))
    assert_size_stride(permute_278, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(getitem_320, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_321, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_323, (8, 8, 64), (512, 64, 1))
    assert_size_stride(getitem_324, (), ())
    assert_size_stride(getitem_325, (), ())
    assert_size_stride(view_414, (392, 512), (512, 1))
    assert_size_stride(mul_242, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_416, (392, 512), (512, 1))
    assert_size_stride(addmm_133, (392, 2048), (2048, 1))
    assert_size_stride(view_418, (392, 2048), (2048, 1))
    assert_size_stride(mul_247, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_420, (392, 512), (512, 1))
    assert_size_stride(permute_286, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(getitem_330, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_331, (8, 8, 49, 64), (50176, 64, 1024, 1))
    assert_size_stride(getitem_333, (8, 8, 64), (512, 64, 1))
    assert_size_stride(getitem_334, (), ())
    assert_size_stride(getitem_335, (), ())
    assert_size_stride(view_427, (392, 512), (512, 1))
    assert_size_stride(mul_249, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_429, (392, 512), (512, 1))
    assert_size_stride(addmm_138, (392, 2048), (2048, 1))
    assert_size_stride(view_431, (392, 2048), (2048, 1))
    assert_size_stride(mul_254, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(clone_166, (8, 512), (512, 1))
    assert_size_stride(permute_294, (1000, 512), (512, 1))
    assert_size_stride(div_1, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_298, (512, 2048), (2048, 1))
    assert_size_stride(permute_302, (2048, 512), (512, 1))
    assert_size_stride(div_2, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_306, (512, 512), (512, 1))
    assert_size_stride(alias_28, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(permute_312, (1024, 512), (512, 1))
    assert_size_stride(permute_317, (512, 512), (512, 1))
    assert_size_stride(div_3, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_321, (512, 2048), (2048, 1))
    assert_size_stride(permute_325, (2048, 512), (512, 1))
    assert_size_stride(div_4, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_329, (512, 512), (512, 1))
    assert_size_stride(alias_29, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(permute_335, (1024, 512), (512, 1))
    assert_size_stride(permute_340, (512, 512), (512, 1))
    assert_size_stride(div_5, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_346, (512, 2048), (2048, 1))
    assert_size_stride(permute_350, (2048, 512), (512, 1))
    assert_size_stride(div_6, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_354, (512, 512), (512, 1))
    assert_size_stride(alias_30, (8, 8, 49, 64), (25088, 1, 512, 8))
    assert_size_stride(permute_360, (1024, 512), (512, 1))
    assert_size_stride(permute_365, (512, 512), (512, 1))
    assert_size_stride(div_7, (8, 49, 1), (49, 1, 1))
    assert_size_stride(div_8, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_371, (320, 1280), (1280, 1))
    assert_size_stride(permute_375, (1280, 320), (320, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_379, (320, 320), (320, 1))
    assert_size_stride(alias_31, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_385, (640, 320), (320, 1))
    assert_size_stride(div_10, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_392, (320, 320), (320, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_396, (320, 1280), (1280, 1))
    assert_size_stride(permute_400, (1280, 320), (320, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_404, (320, 320), (320, 1))
    assert_size_stride(alias_32, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_410, (640, 320), (320, 1))
    assert_size_stride(div_13, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_417, (320, 320), (320, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_421, (320, 1280), (1280, 1))
    assert_size_stride(permute_425, (1280, 320), (320, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_429, (320, 320), (320, 1))
    assert_size_stride(alias_33, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_435, (640, 320), (320, 1))
    assert_size_stride(div_16, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_442, (320, 320), (320, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_446, (320, 1280), (1280, 1))
    assert_size_stride(permute_450, (1280, 320), (320, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_454, (320, 320), (320, 1))
    assert_size_stride(alias_34, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_460, (640, 320), (320, 1))
    assert_size_stride(div_19, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_467, (320, 320), (320, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_471, (320, 1280), (1280, 1))
    assert_size_stride(permute_475, (1280, 320), (320, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_479, (320, 320), (320, 1))
    assert_size_stride(alias_35, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_485, (640, 320), (320, 1))
    assert_size_stride(div_22, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_492, (320, 320), (320, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_496, (320, 1280), (1280, 1))
    assert_size_stride(permute_500, (1280, 320), (320, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_504, (320, 320), (320, 1))
    assert_size_stride(alias_36, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_510, (640, 320), (320, 1))
    assert_size_stride(div_25, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_517, (320, 320), (320, 1))
    assert_size_stride(div_26, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_521, (320, 1280), (1280, 1))
    assert_size_stride(permute_525, (1280, 320), (320, 1))
    assert_size_stride(div_27, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_529, (320, 320), (320, 1))
    assert_size_stride(alias_37, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_535, (640, 320), (320, 1))
    assert_size_stride(div_28, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_542, (320, 320), (320, 1))
    assert_size_stride(div_29, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_546, (320, 1280), (1280, 1))
    assert_size_stride(permute_550, (1280, 320), (320, 1))
    assert_size_stride(div_30, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_554, (320, 320), (320, 1))
    assert_size_stride(alias_38, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_560, (640, 320), (320, 1))
    assert_size_stride(div_31, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_567, (320, 320), (320, 1))
    assert_size_stride(div_32, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_571, (320, 1280), (1280, 1))
    assert_size_stride(permute_575, (1280, 320), (320, 1))
    assert_size_stride(div_33, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_579, (320, 320), (320, 1))
    assert_size_stride(alias_39, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_585, (640, 320), (320, 1))
    assert_size_stride(div_34, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_592, (320, 320), (320, 1))
    assert_size_stride(div_35, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_596, (320, 1280), (1280, 1))
    assert_size_stride(permute_600, (1280, 320), (320, 1))
    assert_size_stride(div_36, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_604, (320, 320), (320, 1))
    assert_size_stride(alias_40, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_610, (640, 320), (320, 1))
    assert_size_stride(div_37, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_617, (320, 320), (320, 1))
    assert_size_stride(div_38, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_621, (320, 1280), (1280, 1))
    assert_size_stride(permute_625, (1280, 320), (320, 1))
    assert_size_stride(div_39, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_629, (320, 320), (320, 1))
    assert_size_stride(alias_41, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_635, (640, 320), (320, 1))
    assert_size_stride(div_40, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_642, (320, 320), (320, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_646, (320, 1280), (1280, 1))
    assert_size_stride(permute_650, (1280, 320), (320, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_654, (320, 320), (320, 1))
    assert_size_stride(alias_42, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_660, (640, 320), (320, 1))
    assert_size_stride(div_43, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_667, (320, 320), (320, 1))
    assert_size_stride(div_44, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_671, (320, 1280), (1280, 1))
    assert_size_stride(permute_675, (1280, 320), (320, 1))
    assert_size_stride(div_45, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_679, (320, 320), (320, 1))
    assert_size_stride(alias_43, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_685, (640, 320), (320, 1))
    assert_size_stride(div_46, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_692, (320, 320), (320, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_696, (320, 1280), (1280, 1))
    assert_size_stride(permute_700, (1280, 320), (320, 1))
    assert_size_stride(div_48, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_704, (320, 320), (320, 1))
    assert_size_stride(alias_44, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_710, (640, 320), (320, 1))
    assert_size_stride(div_49, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_717, (320, 320), (320, 1))
    assert_size_stride(div_50, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_721, (320, 1280), (1280, 1))
    assert_size_stride(permute_725, (1280, 320), (320, 1))
    assert_size_stride(div_51, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_729, (320, 320), (320, 1))
    assert_size_stride(alias_45, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_735, (640, 320), (320, 1))
    assert_size_stride(div_52, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_742, (320, 320), (320, 1))
    assert_size_stride(div_53, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_746, (320, 1280), (1280, 1))
    assert_size_stride(permute_750, (1280, 320), (320, 1))
    assert_size_stride(div_54, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_754, (320, 320), (320, 1))
    assert_size_stride(alias_46, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_760, (640, 320), (320, 1))
    assert_size_stride(div_55, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_767, (320, 320), (320, 1))
    assert_size_stride(div_56, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_771, (320, 1280), (1280, 1))
    assert_size_stride(permute_775, (1280, 320), (320, 1))
    assert_size_stride(div_57, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_779, (320, 320), (320, 1))
    assert_size_stride(alias_47, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_785, (640, 320), (320, 1))
    assert_size_stride(div_58, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_792, (320, 320), (320, 1))
    assert_size_stride(div_59, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_798, (320, 1280), (1280, 1))
    assert_size_stride(permute_802, (1280, 320), (320, 1))
    assert_size_stride(div_60, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_806, (320, 320), (320, 1))
    assert_size_stride(alias_48, (8, 5, 196, 64), (62720, 1, 320, 5))
    assert_size_stride(permute_812, (640, 320), (320, 1))
    assert_size_stride(div_61, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_819, (320, 320), (320, 1))
    assert_size_stride(div_62, (8, 196, 1), (196, 1, 1))
    assert_size_stride(div_63, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_825, (128, 1024), (1024, 1))
    assert_size_stride(permute_829, (1024, 128), (128, 1))
    assert_size_stride(div_64, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_833, (128, 128), (128, 1))
    assert_size_stride(alias_49, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(permute_839, (256, 128), (128, 1))
    assert_size_stride(div_65, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_846, (128, 128), (128, 1))
    assert_size_stride(div_66, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_850, (128, 1024), (1024, 1))
    assert_size_stride(permute_854, (1024, 128), (128, 1))
    assert_size_stride(div_67, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_858, (128, 128), (128, 1))
    assert_size_stride(alias_50, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(permute_864, (256, 128), (128, 1))
    assert_size_stride(div_68, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_871, (128, 128), (128, 1))
    assert_size_stride(div_69, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_875, (128, 1024), (1024, 1))
    assert_size_stride(permute_879, (1024, 128), (128, 1))
    assert_size_stride(div_70, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_883, (128, 128), (128, 1))
    assert_size_stride(alias_51, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(permute_889, (256, 128), (128, 1))
    assert_size_stride(div_71, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_896, (128, 128), (128, 1))
    assert_size_stride(div_72, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_902, (128, 1024), (1024, 1))
    assert_size_stride(permute_906, (1024, 128), (128, 1))
    assert_size_stride(div_73, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_910, (128, 128), (128, 1))
    assert_size_stride(alias_52, (8, 2, 784, 64), (100352, 1, 128, 2))
    assert_size_stride(permute_916, (256, 128), (128, 1))
    assert_size_stride(div_74, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_923, (128, 128), (128, 1))
    assert_size_stride(div_75, (8, 784, 1), (784, 1, 1))
    assert_size_stride(div_76, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_929, (64, 512), (512, 1))
    assert_size_stride(permute_933, (512, 64), (64, 1))
    assert_size_stride(div_77, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_937, (64, 64), (64, 1))
    assert_size_stride(alias_53, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(permute_943, (128, 64), (64, 1))
    assert_size_stride(div_78, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_950, (64, 64), (64, 1))
    assert_size_stride(div_79, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_954, (64, 512), (512, 1))
    assert_size_stride(permute_958, (512, 64), (64, 1))
    assert_size_stride(div_80, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_962, (64, 64), (64, 1))
    assert_size_stride(alias_54, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(permute_968, (128, 64), (64, 1))
    assert_size_stride(div_81, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_975, (64, 64), (64, 1))
    assert_size_stride(div_82, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_981, (64, 512), (512, 1))
    assert_size_stride(permute_985, (512, 64), (64, 1))
    assert_size_stride(div_83, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(permute_989, (64, 64), (64, 1))
    assert_size_stride(alias_55, (8, 1, 3136, 64), (200704, 64, 64, 1))
    assert_size_stride(permute_995, (128, 64), (64, 1))
    assert_size_stride(div_84, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_1002, (64, 64), (64, 1))
    assert_size_stride(div_85, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(div_86, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_294, out=buf0)
        del permute_294
        buf1 = empty((1000, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_166, out=buf1)
        del clone_166
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_1.run(buf0, primals_517, mul_254, div_1, buf5, 392, 512, grid=grid(392), stream=stream0)
        del div_1
        del primals_517
        buf6 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_2.run(buf0, mul_254, buf6, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_254
        buf7 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf6, buf7, 512, 4, grid=grid(512), stream=stream0)
        buf8 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_4.run(buf0, buf8, 512, 392, grid=grid(512), stream=stream0)
        buf9 = empty((392, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (392, 512), (512, 1), 0), permute_298, out=buf9)
        del permute_298
        buf10 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 392), (1, 512), 0), view_431, out=buf10)
        del view_431
        buf11 = reinterpret_tensor(buf6, (1, 512, 4), (2048, 1, 512), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 2048, 98, grid=grid(2048), stream=stream0)
        buf12 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf11, buf12, 512, 4, grid=grid(512), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 49, 2048), (100352, 2048, 1), 0); del buf9  # reuse
        # Source Nodes: [x_467], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_138, 802816, grid=grid(802816), stream=stream0)
        del addmm_138
        buf14 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (392, 2048), (2048, 1), 0), permute_302, out=buf14)
        del permute_302
        buf15 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 392), (1, 2048), 0), view_429, out=buf15)
        del view_429
        buf16 = empty_strided((1, 2048, 4), (8192, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 8192, 98, grid=grid(8192), stream=stream0)
        buf17 = reinterpret_tensor(buf11, (1, 2048), (2048, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 2048, 4, grid=grid(2048), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_511, mul_249, div_2, 392, 512, grid=grid(392), stream=stream0)
        del div_2
        del primals_511
        buf20 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_249, buf20, buf22, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_249
        buf21 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf20, buf21, 512, 4, grid=grid(512), stream=stream0)
        buf23 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf22, buf23, 512, 4, grid=grid(512), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (392, 512), (512, 1), 0), permute_306, out=buf25)
        del permute_306
        buf26 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 392), (1, 512), 0), view_427, out=buf26)
        del view_427
        buf27 = reinterpret_tensor(buf22, (1, 512, 4), (2048, 1, 512), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf24, buf27, 2048, 98, grid=grid(2048), stream=stream0)
        buf28 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf27, buf28, 512, 4, grid=grid(512), stream=stream0)
        buf29 = empty((8, 8, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(permute_286, buf29, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del permute_286
        buf30 = empty((8, 8, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(alias_28, buf30, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del alias_28
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf31 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf25, (8, 8, 49, 64), (25088, 64, 512, 1), 0), buf29, getitem_330, getitem_331, None, buf30, getitem_333, getitem_334, getitem_335, 0.0, [True, True, True, False])
        del buf25
        del buf29
        del buf30
        del getitem_330
        del getitem_331
        del getitem_333
        del getitem_334
        del getitem_335
        buf32 = buf31[0]
        buf33 = buf31[1]
        buf34 = buf31[2]
        del buf31
        buf35 = empty((8, 49, 2, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf33, buf34, buf35, 401408, grid=grid(401408), stream=stream0)
        buf36 = reinterpret_tensor(buf34, (392, 512), (512, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (392, 1024), (1024, 1), 0), permute_312, out=buf36)
        del permute_312
        buf37 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (1024, 392), (1, 1024), 0), view_420, out=buf37)
        buf38 = reinterpret_tensor(buf0, (1, 1024, 4), (4096, 1, 1024), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf35, buf38, 4096, 98, grid=grid(4096), stream=stream0)
        buf39 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf38, buf39, 1024, 4, grid=grid(1024), stream=stream0)
        buf40 = reinterpret_tensor(buf33, (392, 512), (512, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (392, 512), (512, 1), 0), permute_317, out=buf40)
        del permute_317
        buf41 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (512, 392), (1, 512), 0), view_420, out=buf41)
        del view_420
        buf42 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf32, buf42, 2048, 98, grid=grid(2048), stream=stream0)
        buf43 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf42, buf43, 512, 4, grid=grid(512), stream=stream0)
        buf50 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf50, buf36, buf40, primals_503, mul_247, div_3, 392, 512, grid=grid(392), stream=stream0)
        del div_3
        del primals_503
        buf46 = reinterpret_tensor(buf42, (512, 4), (1, 512), 0); del buf42  # reuse
        buf48 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_17.run(buf36, buf40, mul_247, buf46, buf48, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_247
        buf47 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf46, buf47, 512, 4, grid=grid(512), stream=stream0)
        buf49 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf48, buf49, 512, 4, grid=grid(512), stream=stream0)
        buf51 = reinterpret_tensor(buf13, (392, 2048), (2048, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (392, 512), (512, 1), 0), permute_321, out=buf51)
        del permute_321
        buf52 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (512, 392), (1, 512), 0), view_418, out=buf52)
        del view_418
        buf53 = reinterpret_tensor(buf48, (1, 512, 4), (2048, 1, 512), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf50, buf53, 2048, 98, grid=grid(2048), stream=stream0)
        buf54 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf53, buf54, 512, 4, grid=grid(512), stream=stream0)
        buf55 = reinterpret_tensor(buf51, (8, 49, 2048), (100352, 2048, 1), 0); del buf51  # reuse
        # Source Nodes: [x_454], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf55, addmm_133, 802816, grid=grid(802816), stream=stream0)
        del addmm_133
        buf56 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (392, 2048), (2048, 1), 0), permute_325, out=buf56)
        del permute_325
        buf57 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (2048, 392), (1, 2048), 0), view_416, out=buf57)
        del view_416
        buf58 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf55, buf58, 8192, 98, grid=grid(8192), stream=stream0)
        buf59 = reinterpret_tensor(buf53, (1, 2048), (2048, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf58, buf59, 2048, 4, grid=grid(2048), stream=stream0)
        buf66 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf66, buf56, primals_497, mul_242, div_4, 392, 512, grid=grid(392), stream=stream0)
        del div_4
        del primals_497
        buf62 = buf46; del buf46  # reuse
        buf64 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf56, mul_242, buf62, buf64, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_242
        buf63 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf62, buf63, 512, 4, grid=grid(512), stream=stream0)
        buf65 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf64, buf65, 512, 4, grid=grid(512), stream=stream0)
        buf67 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (392, 512), (512, 1), 0), permute_329, out=buf67)
        del permute_329
        buf68 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (512, 392), (1, 512), 0), view_414, out=buf68)
        del view_414
        buf69 = reinterpret_tensor(buf64, (1, 512, 4), (2048, 1, 512), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf66, buf69, 2048, 98, grid=grid(2048), stream=stream0)
        buf70 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf69, buf70, 512, 4, grid=grid(512), stream=stream0)
        buf71 = reinterpret_tensor(buf36, (8, 8, 49, 64), (25088, 3136, 64, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(permute_278, buf71, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del permute_278
        buf72 = reinterpret_tensor(buf32, (8, 8, 49, 64), (25088, 3136, 64, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(alias_29, buf72, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del alias_29
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf73 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf67, (8, 8, 49, 64), (25088, 64, 512, 1), 0), buf71, getitem_320, getitem_321, None, buf72, getitem_323, getitem_324, getitem_325, 0.0, [True, True, True, False])
        del buf67
        del buf71
        del buf72
        del getitem_320
        del getitem_321
        del getitem_323
        del getitem_324
        del getitem_325
        buf74 = buf73[0]
        buf75 = buf73[1]
        buf76 = buf73[2]
        del buf73
        buf77 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf75, buf76, buf77, 401408, grid=grid(401408), stream=stream0)
        buf78 = reinterpret_tensor(buf76, (392, 512), (512, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (392, 1024), (1024, 1), 0), permute_335, out=buf78)
        del permute_335
        buf79 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1024, 392), (1, 1024), 0), view_407, out=buf79)
        buf80 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf77, buf80, 4096, 98, grid=grid(4096), stream=stream0)
        buf81 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf80, buf81, 1024, 4, grid=grid(1024), stream=stream0)
        buf82 = reinterpret_tensor(buf75, (392, 512), (512, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (392, 512), (512, 1), 0), permute_340, out=buf82)
        del permute_340
        buf83 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (512, 392), (1, 512), 0), view_407, out=buf83)
        del view_407
        buf84 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf74, buf84, 2048, 98, grid=grid(2048), stream=stream0)
        del buf74
        buf85 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf84, buf85, 512, 4, grid=grid(512), stream=stream0)
        buf92 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf92, buf78, buf82, primals_489, mul_240, div_5, 392, 512, grid=grid(392), stream=stream0)
        del div_5
        del primals_489
        buf88 = reinterpret_tensor(buf84, (512, 4), (1, 512), 0); del buf84  # reuse
        buf90 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_17.run(buf78, buf82, mul_240, buf88, buf90, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_240
        buf89 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf88, buf89, 512, 4, grid=grid(512), stream=stream0)
        buf91 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf90, buf91, 512, 4, grid=grid(512), stream=stream0)
        buf93 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_5.run(buf92, buf93, 2048, 98, grid=grid(2048), stream=stream0)
        buf94 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf93, buf94, 512, 4, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf95 = aten.convolution_backward(reinterpret_tensor(buf92, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), view_404, primals_487, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
        del primals_487
        del view_404
        buf96 = buf95[0]
        buf97 = buf95[1]
        del buf95
        buf98 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_18.run(buf98, buf92, 4096, 49, grid=grid(4096, 49), stream=stream0)
        buf99 = reinterpret_tensor(buf92, (392, 512), (512, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf98, buf99, 392, 512, grid=grid(392, 512), stream=stream0)
        buf100 = reinterpret_tensor(buf55, (392, 2048), (2048, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf99, permute_346, out=buf100)
        del permute_346
        buf101 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 392), (1, 512), 0), view_402, out=buf101)
        del view_402
        buf102 = reinterpret_tensor(buf93, (1, 512, 4), (2048, 1, 512), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf99, buf102, 2048, 98, grid=grid(2048), stream=stream0)
        buf103 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf102, buf103, 512, 4, grid=grid(512), stream=stream0)
        buf104 = reinterpret_tensor(buf100, (8, 49, 2048), (100352, 2048, 1), 0); del buf100  # reuse
        # Source Nodes: [x_437], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf104, addmm_128, 802816, grid=grid(802816), stream=stream0)
        del addmm_128
        buf105 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (392, 2048), (2048, 1), 0), permute_350, out=buf105)
        del permute_350
        buf106 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (2048, 392), (1, 2048), 0), view_400, out=buf106)
        del view_400
        buf107 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf104, buf107, 8192, 98, grid=grid(8192), stream=stream0)
        buf108 = reinterpret_tensor(buf102, (1, 2048), (2048, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf107, buf108, 2048, 4, grid=grid(2048), stream=stream0)
        buf115 = reinterpret_tensor(buf82, (8, 49, 512), (25088, 512, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf105, primals_481, mul_235, buf98, div_6, buf115, 392, 512, grid=grid(392), stream=stream0)
        del div_6
        del primals_481
        buf111 = buf88; del buf88  # reuse
        buf113 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf105, mul_235, buf111, buf113, 2048, 98, grid=grid(2048), stream=stream0)
        del mul_235
        buf112 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf111, buf112, 512, 4, grid=grid(512), stream=stream0)
        buf114 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf113, buf114, 512, 4, grid=grid(512), stream=stream0)
        buf116 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (392, 512), (512, 1), 0), permute_354, out=buf116)
        del permute_354
        buf117 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 392), (1, 512), 0), view_398, out=buf117)
        del view_398
        buf118 = reinterpret_tensor(buf113, (1, 512, 4), (2048, 1, 512), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf115, buf118, 2048, 98, grid=grid(2048), stream=stream0)
        buf119 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf118, buf119, 512, 4, grid=grid(512), stream=stream0)
        buf120 = reinterpret_tensor(buf98, (8, 8, 49, 64), (25088, 3136, 64, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(permute_267, buf120, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del permute_267
        buf121 = reinterpret_tensor(buf78, (8, 8, 49, 64), (25088, 3136, 64, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_11.run(alias_30, buf121, 64, 3136, grid=grid(64, 3136), stream=stream0)
        del alias_30
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf122 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf116, (8, 8, 49, 64), (25088, 64, 512, 1), 0), buf120, getitem_310, getitem_311, None, buf121, getitem_313, getitem_314, getitem_315, 0.0, [True, True, True, False])
        del buf116
        del buf120
        del buf121
        del getitem_310
        del getitem_311
        del getitem_313
        del getitem_314
        del getitem_315
        buf123 = buf122[0]
        buf124 = buf122[1]
        buf125 = buf122[2]
        del buf122
        buf126 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf124, buf125, buf126, 401408, grid=grid(401408), stream=stream0)
        buf127 = reinterpret_tensor(buf125, (392, 512), (512, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (392, 1024), (1024, 1), 0), permute_360, out=buf127)
        del permute_360
        buf128 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (1024, 392), (1, 1024), 0), view_391, out=buf128)
        buf129 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf126, buf129, 4096, 98, grid=grid(4096), stream=stream0)
        del buf126
        buf130 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf129, buf130, 1024, 4, grid=grid(1024), stream=stream0)
        buf131 = reinterpret_tensor(buf124, (392, 512), (512, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (392, 512), (512, 1), 0), permute_365, out=buf131)
        del permute_365
        buf132 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 392), (1, 512), 0), view_391, out=buf132)
        del view_391
        buf133 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf123, buf133, 2048, 98, grid=grid(2048), stream=stream0)
        buf134 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf133, buf134, 512, 4, grid=grid(512), stream=stream0)
        buf141 = buf115; del buf115  # reuse
        buf148 = reinterpret_tensor(buf123, (8, 49, 512), (25088, 512, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_21.run(buf141, buf127, buf131, primals_473, mul_233, div_7, primals_471, mul_231, div_8, buf148, 392, 512, grid=grid(392), stream=stream0)
        del div_7
        del div_8
        del primals_471
        del primals_473
        buf137 = reinterpret_tensor(buf133, (512, 4), (1, 512), 0); del buf133  # reuse
        buf139 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_17.run(buf127, buf131, mul_233, buf137, buf139, 2048, 98, grid=grid(2048), stream=stream0)
        del buf127
        del buf131
        del mul_233
        buf138 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf137, buf138, 512, 4, grid=grid(512), stream=stream0)
        buf140 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf139, buf140, 512, 4, grid=grid(512), stream=stream0)
        buf144 = buf139; del buf139  # reuse
        buf146 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf141, mul_231, buf144, buf146, 2048, 98, grid=grid(2048), stream=stream0)
        del buf141
        del mul_231
        buf145 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf144, buf145, 512, 4, grid=grid(512), stream=stream0)
        del buf144
        buf147 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf146, buf147, 512, 4, grid=grid(512), stream=stream0)
        buf149 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_5.run(buf148, buf149, 2048, 98, grid=grid(2048), stream=stream0)
        buf150 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf149, buf150, 512, 4, grid=grid(512), stream=stream0)
        del buf149
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(reinterpret_tensor(buf148, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), permute_264, primals_469, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf148
        del permute_264
        del primals_469
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        buf154 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_22.run(buf152, buf154, 1568, 320, grid=grid(1568, 320), stream=stream0)
        buf155 = empty((1568, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, permute_371, out=buf155)
        del permute_371
        buf156 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (320, 1568), (1, 320), 0), view_387, out=buf156)
        del view_387
        buf157 = empty_strided((1, 320, 13), (4160, 1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf154, buf157, 4160, 121, grid=grid(4160), stream=stream0)
        buf158 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf157, buf158, 320, 13, grid=grid(320), stream=stream0)
        buf159 = reinterpret_tensor(buf155, (8, 196, 1280), (250880, 1280, 1), 0); del buf155  # reuse
        # Source Nodes: [x_419], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf159, addmm_123, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_123
        buf160 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (1568, 1280), (1280, 1), 0), permute_375, out=buf160)
        del permute_375
        buf161 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (1280, 1568), (1, 1280), 0), view_385, out=buf161)
        del view_385
        buf162 = empty_strided((1, 1280, 13), (16640, 1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf159, buf162, 16640, 121, grid=grid(16640), stream=stream0)
        buf163 = empty((1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf162, buf163, 1280, 13, grid=grid(1280), stream=stream0)
        buf170 = empty((8, 196, 320), device='cuda', dtype=torch.float32)
        buf171 = empty((1568, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_28.run(buf160, primals_463, mul_226, buf152, div_9, buf170, buf171, 1568, 320, grid=grid(1568), stream=stream0)
        del div_9
        del primals_463
        buf166 = reinterpret_tensor(buf157, (320, 13), (1, 320), 0); del buf157  # reuse
        buf168 = empty_strided((320, 13), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf160, mul_226, buf166, buf168, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_226
        buf167 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf166, buf167, 320, 13, grid=grid(320), stream=stream0)
        buf169 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf168, buf169, 320, 13, grid=grid(320), stream=stream0)
        buf172 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf171, permute_379, out=buf172)
        del permute_379
        buf173 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (320, 1568), (1, 320), 0), view_383, out=buf173)
        del view_383
        buf174 = reinterpret_tensor(buf168, (1, 320, 13), (4160, 1, 320), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf171, buf174, 4160, 121, grid=grid(4160), stream=stream0)
        buf175 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf174, buf175, 320, 13, grid=grid(320), stream=stream0)
        buf176 = reinterpret_tensor(buf171, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_255, buf176, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_255
        buf177 = reinterpret_tensor(buf152, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_31, buf177, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_31
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf178 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf172, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf176, getitem_298, getitem_299, None, buf177, getitem_301, getitem_302, getitem_303, 0.0, [True, True, True, False])
        del buf172
        del buf176
        del getitem_298
        del getitem_299
        del getitem_301
        del getitem_302
        del getitem_303
        buf179 = buf178[0]
        buf180 = buf178[1]
        buf181 = buf178[2]
        del buf178
        buf182 = empty((8, 49, 2, 5, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf180, buf181, buf182, 250880, grid=grid(250880), stream=stream0)
        buf183 = reinterpret_tensor(buf181, (392, 320), (320, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (392, 640), (640, 1), 0), permute_385, out=buf183)
        del permute_385
        buf184 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (640, 392), (1, 640), 0), view_379, out=buf184)
        del view_379
        buf185 = empty_strided((1, 640, 4), (2560, 1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf182, buf185, 2560, 98, grid=grid(2560), stream=stream0)
        buf186 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf185, buf186, 640, 4, grid=grid(640), stream=stream0)
        buf193 = reinterpret_tensor(buf180, (8, 49, 320), (15680, 320, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf183, primals_457, mul_224, div_10, buf193, 392, 320, grid=grid(392), stream=stream0)
        del div_10
        del primals_457
        buf189 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        buf191 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf183, mul_224, buf189, buf191, 1280, 98, grid=grid(1280), stream=stream0)
        del buf183
        del mul_224
        buf190 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf189, buf190, 320, 4, grid=grid(320), stream=stream0)
        buf192 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf191, buf192, 320, 4, grid=grid(320), stream=stream0)
        buf194 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf193, buf194, 1280, 98, grid=grid(1280), stream=stream0)
        buf195 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf194, buf195, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf196 = aten.convolution_backward(reinterpret_tensor(buf193, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_377, primals_455, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf193
        del primals_455
        del view_377
        buf197 = buf196[0]
        buf198 = buf196[1]
        del buf196
        buf199 = reinterpret_tensor(buf177, (1568, 320), (320, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (1568, 320), (320, 1), 0), permute_392, out=buf199)
        del permute_392
        buf200 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (320, 1568), (1, 320), 0), view_374, out=buf200)
        del view_374
        buf201 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf179, buf201, 4160, 121, grid=grid(4160), stream=stream0)
        buf202 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf201, buf202, 320, 13, grid=grid(320), stream=stream0)
        buf203 = empty_strided((8, 196, 1, 3), (588, 3, 4704, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf197, buf199, primals_451, buf203, 4704, 107, grid=grid(4704), stream=stream0)
        buf204 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf203, buf204, 1568, 3, grid=grid(1568), stream=stream0)
        buf210 = buf170; del buf170  # reuse
        buf211 = reinterpret_tensor(buf179, (1568, 320), (320, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf210, buf197, buf199, primals_451, mul_222, div_11, buf204, buf211, 1568, 320, grid=grid(1568), stream=stream0)
        del div_11
        del primals_451
        buf206 = reinterpret_tensor(buf201, (320, 13), (1, 320), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf197, buf199, mul_222, buf206, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_222
        buf207 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf206, buf207, 320, 13, grid=grid(320), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (320, 13), (13, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf197, buf199, buf208, 4160, 121, grid=grid(4160), stream=stream0)
        buf209 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf208, buf209, 320, 13, grid=grid(320), stream=stream0)
        buf212 = reinterpret_tensor(buf159, (1568, 1280), (1280, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf211, permute_396, out=buf212)
        del permute_396
        buf213 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (320, 1568), (1, 320), 0), view_372, out=buf213)
        del view_372
        buf214 = reinterpret_tensor(buf208, (1, 320, 13), (4160, 1, 320), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf211, buf214, 4160, 121, grid=grid(4160), stream=stream0)
        buf215 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf214, buf215, 320, 13, grid=grid(320), stream=stream0)
        buf216 = reinterpret_tensor(buf212, (8, 196, 1280), (250880, 1280, 1), 0); del buf212  # reuse
        # Source Nodes: [x_403], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf216, addmm_118, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_118
        buf217 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (1568, 1280), (1280, 1), 0), permute_400, out=buf217)
        del permute_400
        buf218 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (1280, 1568), (1, 1280), 0), view_370, out=buf218)
        del view_370
        buf219 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf216, buf219, 16640, 121, grid=grid(16640), stream=stream0)
        buf220 = reinterpret_tensor(buf194, (1, 1280), (1280, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf219, buf220, 1280, 13, grid=grid(1280), stream=stream0)
        buf227 = buf210; del buf210  # reuse
        buf228 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf227, buf217, primals_445, mul_217, div_12, buf228, 1568, 320, grid=grid(1568), stream=stream0)
        del div_12
        del primals_445
        buf223 = reinterpret_tensor(buf214, (320, 13), (1, 320), 0); del buf214  # reuse
        buf225 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf217, mul_217, buf223, buf225, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_217
        buf224 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf223, buf224, 320, 13, grid=grid(320), stream=stream0)
        buf226 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf225, buf226, 320, 13, grid=grid(320), stream=stream0)
        buf229 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf228, permute_404, out=buf229)
        del permute_404
        buf230 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (320, 1568), (1, 320), 0), view_368, out=buf230)
        del view_368
        buf231 = reinterpret_tensor(buf225, (1, 320, 13), (4160, 1, 320), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf228, buf231, 4160, 121, grid=grid(4160), stream=stream0)
        buf232 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf231, buf232, 320, 13, grid=grid(320), stream=stream0)
        buf233 = reinterpret_tensor(buf228, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_245, buf233, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_245
        buf234 = reinterpret_tensor(buf197, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_32, buf234, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_32
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf235 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf229, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf233, getitem_286, getitem_287, None, buf234, getitem_289, getitem_290, getitem_291, 0.0, [True, True, True, False])
        del buf229
        del buf233
        del getitem_286
        del getitem_287
        del getitem_289
        del getitem_290
        del getitem_291
        buf236 = buf235[0]
        buf237 = buf235[1]
        buf238 = buf235[2]
        del buf235
        buf239 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf237, buf238, buf239, 250880, grid=grid(250880), stream=stream0)
        buf240 = reinterpret_tensor(buf238, (392, 320), (320, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (392, 640), (640, 1), 0), permute_410, out=buf240)
        del permute_410
        buf241 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (640, 392), (1, 640), 0), view_364, out=buf241)
        del view_364
        buf242 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf239, buf242, 2560, 98, grid=grid(2560), stream=stream0)
        buf243 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf242, buf243, 640, 4, grid=grid(640), stream=stream0)
        buf250 = reinterpret_tensor(buf237, (8, 49, 320), (15680, 320, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf240, primals_439, mul_215, div_13, buf250, 392, 320, grid=grid(392), stream=stream0)
        del div_13
        del primals_439
        buf246 = buf189; del buf189  # reuse
        buf248 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf240, mul_215, buf246, buf248, 1280, 98, grid=grid(1280), stream=stream0)
        del buf240
        del mul_215
        buf247 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf246, buf247, 320, 4, grid=grid(320), stream=stream0)
        buf249 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf248, buf249, 320, 4, grid=grid(320), stream=stream0)
        buf251 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf250, buf251, 1280, 98, grid=grid(1280), stream=stream0)
        buf252 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf251, buf252, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf253 = aten.convolution_backward(reinterpret_tensor(buf250, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_362, primals_437, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf250
        del primals_437
        del view_362
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        buf256 = reinterpret_tensor(buf234, (1568, 320), (320, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (1568, 320), (320, 1), 0), permute_417, out=buf256)
        del permute_417
        buf257 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (320, 1568), (1, 320), 0), view_359, out=buf257)
        del view_359
        buf258 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf236, buf258, 4160, 121, grid=grid(4160), stream=stream0)
        buf259 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf258, buf259, 320, 13, grid=grid(320), stream=stream0)
        buf260 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf254, buf256, primals_433, buf260, 4704, 107, grid=grid(4704), stream=stream0)
        buf261 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf260, buf261, 1568, 3, grid=grid(1568), stream=stream0)
        buf267 = buf227; del buf227  # reuse
        buf268 = reinterpret_tensor(buf236, (1568, 320), (320, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf267, buf254, buf256, primals_433, mul_213, div_14, buf261, buf268, 1568, 320, grid=grid(1568), stream=stream0)
        del div_14
        del primals_433
        buf263 = reinterpret_tensor(buf258, (320, 13), (1, 320), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf254, buf256, mul_213, buf263, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_213
        buf264 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf263, buf264, 320, 13, grid=grid(320), stream=stream0)
        buf265 = reinterpret_tensor(buf263, (320, 13), (13, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf254, buf256, buf265, 4160, 121, grid=grid(4160), stream=stream0)
        buf266 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf265, buf266, 320, 13, grid=grid(320), stream=stream0)
        buf269 = reinterpret_tensor(buf216, (1568, 1280), (1280, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, permute_421, out=buf269)
        del permute_421
        buf270 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (320, 1568), (1, 320), 0), view_357, out=buf270)
        del view_357
        buf271 = reinterpret_tensor(buf265, (1, 320, 13), (4160, 1, 320), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf268, buf271, 4160, 121, grid=grid(4160), stream=stream0)
        buf272 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf271, buf272, 320, 13, grid=grid(320), stream=stream0)
        buf273 = reinterpret_tensor(buf269, (8, 196, 1280), (250880, 1280, 1), 0); del buf269  # reuse
        # Source Nodes: [x_387], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf273, addmm_113, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_113
        buf274 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (1568, 1280), (1280, 1), 0), permute_425, out=buf274)
        del permute_425
        buf275 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (1280, 1568), (1, 1280), 0), view_355, out=buf275)
        del view_355
        buf276 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf273, buf276, 16640, 121, grid=grid(16640), stream=stream0)
        buf277 = reinterpret_tensor(buf251, (1, 1280), (1280, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf276, buf277, 1280, 13, grid=grid(1280), stream=stream0)
        buf284 = buf267; del buf267  # reuse
        buf285 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf284, buf274, primals_427, mul_208, div_15, buf285, 1568, 320, grid=grid(1568), stream=stream0)
        del div_15
        del primals_427
        buf280 = reinterpret_tensor(buf271, (320, 13), (1, 320), 0); del buf271  # reuse
        buf282 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf274, mul_208, buf280, buf282, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_208
        buf281 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf280, buf281, 320, 13, grid=grid(320), stream=stream0)
        buf283 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf282, buf283, 320, 13, grid=grid(320), stream=stream0)
        buf286 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf285, permute_429, out=buf286)
        del permute_429
        buf287 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (320, 1568), (1, 320), 0), view_353, out=buf287)
        del view_353
        buf288 = reinterpret_tensor(buf282, (1, 320, 13), (4160, 1, 320), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf285, buf288, 4160, 121, grid=grid(4160), stream=stream0)
        buf289 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf288, buf289, 320, 13, grid=grid(320), stream=stream0)
        buf290 = reinterpret_tensor(buf285, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_235, buf290, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_235
        buf291 = reinterpret_tensor(buf254, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_33, buf291, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_33
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf292 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf286, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf290, getitem_274, getitem_275, None, buf291, getitem_277, getitem_278, getitem_279, 0.0, [True, True, True, False])
        del buf286
        del buf290
        del getitem_274
        del getitem_275
        del getitem_277
        del getitem_278
        del getitem_279
        buf293 = buf292[0]
        buf294 = buf292[1]
        buf295 = buf292[2]
        del buf292
        buf296 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf294, buf295, buf296, 250880, grid=grid(250880), stream=stream0)
        buf297 = reinterpret_tensor(buf295, (392, 320), (320, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (392, 640), (640, 1), 0), permute_435, out=buf297)
        del permute_435
        buf298 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (640, 392), (1, 640), 0), view_349, out=buf298)
        del view_349
        buf299 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf296, buf299, 2560, 98, grid=grid(2560), stream=stream0)
        buf300 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf299, buf300, 640, 4, grid=grid(640), stream=stream0)
        buf307 = reinterpret_tensor(buf294, (8, 49, 320), (15680, 320, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf297, primals_421, mul_206, div_16, buf307, 392, 320, grid=grid(392), stream=stream0)
        del div_16
        del primals_421
        buf303 = buf246; del buf246  # reuse
        buf305 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf297, mul_206, buf303, buf305, 1280, 98, grid=grid(1280), stream=stream0)
        del buf297
        del mul_206
        buf304 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf303, buf304, 320, 4, grid=grid(320), stream=stream0)
        buf306 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf305, buf306, 320, 4, grid=grid(320), stream=stream0)
        buf308 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf307, buf308, 1280, 98, grid=grid(1280), stream=stream0)
        buf309 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf308, buf309, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf310 = aten.convolution_backward(reinterpret_tensor(buf307, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_347, primals_419, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf307
        del primals_419
        del view_347
        buf311 = buf310[0]
        buf312 = buf310[1]
        del buf310
        buf313 = reinterpret_tensor(buf291, (1568, 320), (320, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (1568, 320), (320, 1), 0), permute_442, out=buf313)
        del permute_442
        buf314 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (320, 1568), (1, 320), 0), view_344, out=buf314)
        del view_344
        buf315 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf293, buf315, 4160, 121, grid=grid(4160), stream=stream0)
        buf316 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf315, buf316, 320, 13, grid=grid(320), stream=stream0)
        buf317 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf311, buf313, primals_415, buf317, 4704, 107, grid=grid(4704), stream=stream0)
        buf318 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf317, buf318, 1568, 3, grid=grid(1568), stream=stream0)
        buf324 = buf284; del buf284  # reuse
        buf325 = reinterpret_tensor(buf293, (1568, 320), (320, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf324, buf311, buf313, primals_415, mul_204, div_17, buf318, buf325, 1568, 320, grid=grid(1568), stream=stream0)
        del div_17
        del primals_415
        buf320 = reinterpret_tensor(buf315, (320, 13), (1, 320), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf311, buf313, mul_204, buf320, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_204
        buf321 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf320, buf321, 320, 13, grid=grid(320), stream=stream0)
        buf322 = reinterpret_tensor(buf320, (320, 13), (13, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf311, buf313, buf322, 4160, 121, grid=grid(4160), stream=stream0)
        buf323 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf322, buf323, 320, 13, grid=grid(320), stream=stream0)
        buf326 = reinterpret_tensor(buf273, (1568, 1280), (1280, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf325, permute_446, out=buf326)
        del permute_446
        buf327 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (320, 1568), (1, 320), 0), view_342, out=buf327)
        del view_342
        buf328 = reinterpret_tensor(buf322, (1, 320, 13), (4160, 1, 320), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf325, buf328, 4160, 121, grid=grid(4160), stream=stream0)
        buf329 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf328, buf329, 320, 13, grid=grid(320), stream=stream0)
        buf330 = reinterpret_tensor(buf326, (8, 196, 1280), (250880, 1280, 1), 0); del buf326  # reuse
        # Source Nodes: [x_371], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf330, addmm_108, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_108
        buf331 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (1568, 1280), (1280, 1), 0), permute_450, out=buf331)
        del permute_450
        buf332 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (1280, 1568), (1, 1280), 0), view_340, out=buf332)
        del view_340
        buf333 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf330, buf333, 16640, 121, grid=grid(16640), stream=stream0)
        buf334 = reinterpret_tensor(buf308, (1, 1280), (1280, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf333, buf334, 1280, 13, grid=grid(1280), stream=stream0)
        buf341 = buf324; del buf324  # reuse
        buf342 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf341, buf331, primals_409, mul_199, div_18, buf342, 1568, 320, grid=grid(1568), stream=stream0)
        del div_18
        del primals_409
        buf337 = reinterpret_tensor(buf328, (320, 13), (1, 320), 0); del buf328  # reuse
        buf339 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf331, mul_199, buf337, buf339, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_199
        buf338 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf337, buf338, 320, 13, grid=grid(320), stream=stream0)
        buf340 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf339, buf340, 320, 13, grid=grid(320), stream=stream0)
        buf343 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf342, permute_454, out=buf343)
        del permute_454
        buf344 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (320, 1568), (1, 320), 0), view_338, out=buf344)
        del view_338
        buf345 = reinterpret_tensor(buf339, (1, 320, 13), (4160, 1, 320), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf342, buf345, 4160, 121, grid=grid(4160), stream=stream0)
        buf346 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf345, buf346, 320, 13, grid=grid(320), stream=stream0)
        buf347 = reinterpret_tensor(buf342, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_225, buf347, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_225
        buf348 = reinterpret_tensor(buf311, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_34, buf348, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_34
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf349 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf343, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf347, getitem_262, getitem_263, None, buf348, getitem_265, getitem_266, getitem_267, 0.0, [True, True, True, False])
        del buf343
        del buf347
        del getitem_262
        del getitem_263
        del getitem_265
        del getitem_266
        del getitem_267
        buf350 = buf349[0]
        buf351 = buf349[1]
        buf352 = buf349[2]
        del buf349
        buf353 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf351, buf352, buf353, 250880, grid=grid(250880), stream=stream0)
        buf354 = reinterpret_tensor(buf352, (392, 320), (320, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (392, 640), (640, 1), 0), permute_460, out=buf354)
        del permute_460
        buf355 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (640, 392), (1, 640), 0), view_334, out=buf355)
        del view_334
        buf356 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf353, buf356, 2560, 98, grid=grid(2560), stream=stream0)
        buf357 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf356, buf357, 640, 4, grid=grid(640), stream=stream0)
        buf364 = reinterpret_tensor(buf351, (8, 49, 320), (15680, 320, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf354, primals_403, mul_197, div_19, buf364, 392, 320, grid=grid(392), stream=stream0)
        del div_19
        del primals_403
        buf360 = buf303; del buf303  # reuse
        buf362 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf354, mul_197, buf360, buf362, 1280, 98, grid=grid(1280), stream=stream0)
        del buf354
        del mul_197
        buf361 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf360, buf361, 320, 4, grid=grid(320), stream=stream0)
        buf363 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf362, buf363, 320, 4, grid=grid(320), stream=stream0)
        buf365 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf364, buf365, 1280, 98, grid=grid(1280), stream=stream0)
        buf366 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf365, buf366, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf367 = aten.convolution_backward(reinterpret_tensor(buf364, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_332, primals_401, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf364
        del primals_401
        del view_332
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf370 = reinterpret_tensor(buf348, (1568, 320), (320, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (1568, 320), (320, 1), 0), permute_467, out=buf370)
        del permute_467
        buf371 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (320, 1568), (1, 320), 0), view_329, out=buf371)
        del view_329
        buf372 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf350, buf372, 4160, 121, grid=grid(4160), stream=stream0)
        buf373 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf372, buf373, 320, 13, grid=grid(320), stream=stream0)
        buf374 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf368, buf370, primals_397, buf374, 4704, 107, grid=grid(4704), stream=stream0)
        buf375 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf374, buf375, 1568, 3, grid=grid(1568), stream=stream0)
        buf381 = buf341; del buf341  # reuse
        buf382 = reinterpret_tensor(buf350, (1568, 320), (320, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf381, buf368, buf370, primals_397, mul_195, div_20, buf375, buf382, 1568, 320, grid=grid(1568), stream=stream0)
        del div_20
        del primals_397
        buf377 = reinterpret_tensor(buf372, (320, 13), (1, 320), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf368, buf370, mul_195, buf377, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_195
        buf378 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf377, buf378, 320, 13, grid=grid(320), stream=stream0)
        buf379 = reinterpret_tensor(buf377, (320, 13), (13, 1), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf368, buf370, buf379, 4160, 121, grid=grid(4160), stream=stream0)
        buf380 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf379, buf380, 320, 13, grid=grid(320), stream=stream0)
        buf383 = reinterpret_tensor(buf330, (1568, 1280), (1280, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf382, permute_471, out=buf383)
        del permute_471
        buf384 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (320, 1568), (1, 320), 0), view_327, out=buf384)
        del view_327
        buf385 = reinterpret_tensor(buf379, (1, 320, 13), (4160, 1, 320), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf382, buf385, 4160, 121, grid=grid(4160), stream=stream0)
        buf386 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf385, buf386, 320, 13, grid=grid(320), stream=stream0)
        buf387 = reinterpret_tensor(buf383, (8, 196, 1280), (250880, 1280, 1), 0); del buf383  # reuse
        # Source Nodes: [x_355], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf387, addmm_103, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_103
        buf388 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1568, 1280), (1280, 1), 0), permute_475, out=buf388)
        del permute_475
        buf389 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1280, 1568), (1, 1280), 0), view_325, out=buf389)
        del view_325
        buf390 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf387, buf390, 16640, 121, grid=grid(16640), stream=stream0)
        buf391 = reinterpret_tensor(buf365, (1, 1280), (1280, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf390, buf391, 1280, 13, grid=grid(1280), stream=stream0)
        buf398 = buf381; del buf381  # reuse
        buf399 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf398, buf388, primals_391, mul_190, div_21, buf399, 1568, 320, grid=grid(1568), stream=stream0)
        del div_21
        del primals_391
        buf394 = reinterpret_tensor(buf385, (320, 13), (1, 320), 0); del buf385  # reuse
        buf396 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf388, mul_190, buf394, buf396, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_190
        buf395 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf394, buf395, 320, 13, grid=grid(320), stream=stream0)
        buf397 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf396, buf397, 320, 13, grid=grid(320), stream=stream0)
        buf400 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, permute_479, out=buf400)
        del permute_479
        buf401 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (320, 1568), (1, 320), 0), view_323, out=buf401)
        del view_323
        buf402 = reinterpret_tensor(buf396, (1, 320, 13), (4160, 1, 320), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf399, buf402, 4160, 121, grid=grid(4160), stream=stream0)
        buf403 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf402, buf403, 320, 13, grid=grid(320), stream=stream0)
        buf404 = reinterpret_tensor(buf399, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_215, buf404, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_215
        buf405 = reinterpret_tensor(buf368, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_35, buf405, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_35
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf406 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf400, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf404, getitem_250, getitem_251, None, buf405, getitem_253, getitem_254, getitem_255, 0.0, [True, True, True, False])
        del buf400
        del buf404
        del getitem_250
        del getitem_251
        del getitem_253
        del getitem_254
        del getitem_255
        buf407 = buf406[0]
        buf408 = buf406[1]
        buf409 = buf406[2]
        del buf406
        buf410 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf408, buf409, buf410, 250880, grid=grid(250880), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (392, 320), (320, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (392, 640), (640, 1), 0), permute_485, out=buf411)
        del permute_485
        buf412 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (640, 392), (1, 640), 0), view_319, out=buf412)
        del view_319
        buf413 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf410, buf413, 2560, 98, grid=grid(2560), stream=stream0)
        buf414 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf413, buf414, 640, 4, grid=grid(640), stream=stream0)
        buf421 = reinterpret_tensor(buf408, (8, 49, 320), (15680, 320, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf411, primals_385, mul_188, div_22, buf421, 392, 320, grid=grid(392), stream=stream0)
        del div_22
        del primals_385
        buf417 = buf360; del buf360  # reuse
        buf419 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf411, mul_188, buf417, buf419, 1280, 98, grid=grid(1280), stream=stream0)
        del buf411
        del mul_188
        buf418 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf417, buf418, 320, 4, grid=grid(320), stream=stream0)
        buf420 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf419, buf420, 320, 4, grid=grid(320), stream=stream0)
        buf422 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf421, buf422, 1280, 98, grid=grid(1280), stream=stream0)
        buf423 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf422, buf423, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf424 = aten.convolution_backward(reinterpret_tensor(buf421, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_317, primals_383, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf421
        del primals_383
        del view_317
        buf425 = buf424[0]
        buf426 = buf424[1]
        del buf424
        buf427 = reinterpret_tensor(buf405, (1568, 320), (320, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (1568, 320), (320, 1), 0), permute_492, out=buf427)
        del permute_492
        buf428 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (320, 1568), (1, 320), 0), view_314, out=buf428)
        del view_314
        buf429 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf407, buf429, 4160, 121, grid=grid(4160), stream=stream0)
        buf430 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf429, buf430, 320, 13, grid=grid(320), stream=stream0)
        buf431 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf425, buf427, primals_379, buf431, 4704, 107, grid=grid(4704), stream=stream0)
        buf432 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf431, buf432, 1568, 3, grid=grid(1568), stream=stream0)
        buf438 = buf398; del buf398  # reuse
        buf439 = reinterpret_tensor(buf407, (1568, 320), (320, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf438, buf425, buf427, primals_379, mul_186, div_23, buf432, buf439, 1568, 320, grid=grid(1568), stream=stream0)
        del div_23
        del primals_379
        buf434 = reinterpret_tensor(buf429, (320, 13), (1, 320), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf425, buf427, mul_186, buf434, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_186
        buf435 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf434, buf435, 320, 13, grid=grid(320), stream=stream0)
        buf436 = reinterpret_tensor(buf434, (320, 13), (13, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf425, buf427, buf436, 4160, 121, grid=grid(4160), stream=stream0)
        buf437 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf436, buf437, 320, 13, grid=grid(320), stream=stream0)
        buf440 = reinterpret_tensor(buf387, (1568, 1280), (1280, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf439, permute_496, out=buf440)
        del permute_496
        buf441 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (320, 1568), (1, 320), 0), view_312, out=buf441)
        del view_312
        buf442 = reinterpret_tensor(buf436, (1, 320, 13), (4160, 1, 320), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf439, buf442, 4160, 121, grid=grid(4160), stream=stream0)
        buf443 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf442, buf443, 320, 13, grid=grid(320), stream=stream0)
        buf444 = reinterpret_tensor(buf440, (8, 196, 1280), (250880, 1280, 1), 0); del buf440  # reuse
        # Source Nodes: [x_339], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf444, addmm_98, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_98
        buf445 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (1568, 1280), (1280, 1), 0), permute_500, out=buf445)
        del permute_500
        buf446 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (1280, 1568), (1, 1280), 0), view_310, out=buf446)
        del view_310
        buf447 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf444, buf447, 16640, 121, grid=grid(16640), stream=stream0)
        buf448 = reinterpret_tensor(buf422, (1, 1280), (1280, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf447, buf448, 1280, 13, grid=grid(1280), stream=stream0)
        buf455 = buf438; del buf438  # reuse
        buf456 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf455, buf445, primals_373, mul_181, div_24, buf456, 1568, 320, grid=grid(1568), stream=stream0)
        del div_24
        del primals_373
        buf451 = reinterpret_tensor(buf442, (320, 13), (1, 320), 0); del buf442  # reuse
        buf453 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf445, mul_181, buf451, buf453, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_181
        buf452 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf451, buf452, 320, 13, grid=grid(320), stream=stream0)
        buf454 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf453, buf454, 320, 13, grid=grid(320), stream=stream0)
        buf457 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf456, permute_504, out=buf457)
        del permute_504
        buf458 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (320, 1568), (1, 320), 0), view_308, out=buf458)
        del view_308
        buf459 = reinterpret_tensor(buf453, (1, 320, 13), (4160, 1, 320), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf456, buf459, 4160, 121, grid=grid(4160), stream=stream0)
        buf460 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf459, buf460, 320, 13, grid=grid(320), stream=stream0)
        buf461 = reinterpret_tensor(buf456, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_205, buf461, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_205
        buf462 = reinterpret_tensor(buf425, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_36, buf462, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_36
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf463 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf457, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf461, getitem_238, getitem_239, None, buf462, getitem_241, getitem_242, getitem_243, 0.0, [True, True, True, False])
        del buf457
        del buf461
        del getitem_238
        del getitem_239
        del getitem_241
        del getitem_242
        del getitem_243
        buf464 = buf463[0]
        buf465 = buf463[1]
        buf466 = buf463[2]
        del buf463
        buf467 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf465, buf466, buf467, 250880, grid=grid(250880), stream=stream0)
        buf468 = reinterpret_tensor(buf466, (392, 320), (320, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf467, (392, 640), (640, 1), 0), permute_510, out=buf468)
        del permute_510
        buf469 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf467, (640, 392), (1, 640), 0), view_304, out=buf469)
        del view_304
        buf470 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf467, buf470, 2560, 98, grid=grid(2560), stream=stream0)
        buf471 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf470, buf471, 640, 4, grid=grid(640), stream=stream0)
        buf478 = reinterpret_tensor(buf465, (8, 49, 320), (15680, 320, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf468, primals_367, mul_179, div_25, buf478, 392, 320, grid=grid(392), stream=stream0)
        del div_25
        del primals_367
        buf474 = buf417; del buf417  # reuse
        buf476 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf468, mul_179, buf474, buf476, 1280, 98, grid=grid(1280), stream=stream0)
        del buf468
        del mul_179
        buf475 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf474, buf475, 320, 4, grid=grid(320), stream=stream0)
        buf477 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf476, buf477, 320, 4, grid=grid(320), stream=stream0)
        buf479 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf478, buf479, 1280, 98, grid=grid(1280), stream=stream0)
        buf480 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf479, buf480, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = aten.convolution_backward(reinterpret_tensor(buf478, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_302, primals_365, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf478
        del primals_365
        del view_302
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        buf484 = reinterpret_tensor(buf462, (1568, 320), (320, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (1568, 320), (320, 1), 0), permute_517, out=buf484)
        del permute_517
        buf485 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (320, 1568), (1, 320), 0), view_299, out=buf485)
        del view_299
        buf486 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf464, buf486, 4160, 121, grid=grid(4160), stream=stream0)
        buf487 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf486, buf487, 320, 13, grid=grid(320), stream=stream0)
        buf488 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf482, buf484, primals_361, buf488, 4704, 107, grid=grid(4704), stream=stream0)
        buf489 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf488, buf489, 1568, 3, grid=grid(1568), stream=stream0)
        buf495 = buf455; del buf455  # reuse
        buf496 = reinterpret_tensor(buf464, (1568, 320), (320, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf495, buf482, buf484, primals_361, mul_177, div_26, buf489, buf496, 1568, 320, grid=grid(1568), stream=stream0)
        del div_26
        del primals_361
        buf491 = reinterpret_tensor(buf486, (320, 13), (1, 320), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf482, buf484, mul_177, buf491, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_177
        buf492 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf491, buf492, 320, 13, grid=grid(320), stream=stream0)
        buf493 = reinterpret_tensor(buf491, (320, 13), (13, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf482, buf484, buf493, 4160, 121, grid=grid(4160), stream=stream0)
        buf494 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf493, buf494, 320, 13, grid=grid(320), stream=stream0)
        buf497 = reinterpret_tensor(buf444, (1568, 1280), (1280, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf496, permute_521, out=buf497)
        del permute_521
        buf498 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (320, 1568), (1, 320), 0), view_297, out=buf498)
        del view_297
        buf499 = reinterpret_tensor(buf493, (1, 320, 13), (4160, 1, 320), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf496, buf499, 4160, 121, grid=grid(4160), stream=stream0)
        buf500 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf499, buf500, 320, 13, grid=grid(320), stream=stream0)
        buf501 = reinterpret_tensor(buf497, (8, 196, 1280), (250880, 1280, 1), 0); del buf497  # reuse
        # Source Nodes: [x_323], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf501, addmm_93, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_93
        buf502 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1568, 1280), (1280, 1), 0), permute_525, out=buf502)
        del permute_525
        buf503 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1280, 1568), (1, 1280), 0), view_295, out=buf503)
        del view_295
        buf504 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf501, buf504, 16640, 121, grid=grid(16640), stream=stream0)
        buf505 = reinterpret_tensor(buf479, (1, 1280), (1280, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf504, buf505, 1280, 13, grid=grid(1280), stream=stream0)
        buf512 = buf495; del buf495  # reuse
        buf513 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf512, buf502, primals_355, mul_172, div_27, buf513, 1568, 320, grid=grid(1568), stream=stream0)
        del div_27
        del primals_355
        buf508 = reinterpret_tensor(buf499, (320, 13), (1, 320), 0); del buf499  # reuse
        buf510 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf502, mul_172, buf508, buf510, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_172
        buf509 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf508, buf509, 320, 13, grid=grid(320), stream=stream0)
        buf511 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf510, buf511, 320, 13, grid=grid(320), stream=stream0)
        buf514 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf513, permute_529, out=buf514)
        del permute_529
        buf515 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf513, (320, 1568), (1, 320), 0), view_293, out=buf515)
        del view_293
        buf516 = reinterpret_tensor(buf510, (1, 320, 13), (4160, 1, 320), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf513, buf516, 4160, 121, grid=grid(4160), stream=stream0)
        buf517 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf516, buf517, 320, 13, grid=grid(320), stream=stream0)
        buf518 = reinterpret_tensor(buf513, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_195, buf518, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_195
        buf519 = reinterpret_tensor(buf482, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_37, buf519, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_37
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf520 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf514, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf518, getitem_226, getitem_227, None, buf519, getitem_229, getitem_230, getitem_231, 0.0, [True, True, True, False])
        del buf514
        del buf518
        del getitem_226
        del getitem_227
        del getitem_229
        del getitem_230
        del getitem_231
        buf521 = buf520[0]
        buf522 = buf520[1]
        buf523 = buf520[2]
        del buf520
        buf524 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf522, buf523, buf524, 250880, grid=grid(250880), stream=stream0)
        buf525 = reinterpret_tensor(buf523, (392, 320), (320, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf524, (392, 640), (640, 1), 0), permute_535, out=buf525)
        del permute_535
        buf526 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf524, (640, 392), (1, 640), 0), view_289, out=buf526)
        del view_289
        buf527 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf524, buf527, 2560, 98, grid=grid(2560), stream=stream0)
        buf528 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf527, buf528, 640, 4, grid=grid(640), stream=stream0)
        buf535 = reinterpret_tensor(buf522, (8, 49, 320), (15680, 320, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf525, primals_349, mul_170, div_28, buf535, 392, 320, grid=grid(392), stream=stream0)
        del div_28
        del primals_349
        buf531 = buf474; del buf474  # reuse
        buf533 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf525, mul_170, buf531, buf533, 1280, 98, grid=grid(1280), stream=stream0)
        del buf525
        del mul_170
        buf532 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf531, buf532, 320, 4, grid=grid(320), stream=stream0)
        buf534 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf533, buf534, 320, 4, grid=grid(320), stream=stream0)
        buf536 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf535, buf536, 1280, 98, grid=grid(1280), stream=stream0)
        buf537 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf536, buf537, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf538 = aten.convolution_backward(reinterpret_tensor(buf535, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_287, primals_347, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf535
        del primals_347
        del view_287
        buf539 = buf538[0]
        buf540 = buf538[1]
        del buf538
        buf541 = reinterpret_tensor(buf519, (1568, 320), (320, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (1568, 320), (320, 1), 0), permute_542, out=buf541)
        del permute_542
        buf542 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (320, 1568), (1, 320), 0), view_284, out=buf542)
        del view_284
        buf543 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf521, buf543, 4160, 121, grid=grid(4160), stream=stream0)
        buf544 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf543, buf544, 320, 13, grid=grid(320), stream=stream0)
        buf545 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf539, buf541, primals_343, buf545, 4704, 107, grid=grid(4704), stream=stream0)
        buf546 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf545, buf546, 1568, 3, grid=grid(1568), stream=stream0)
        buf552 = buf512; del buf512  # reuse
        buf553 = reinterpret_tensor(buf521, (1568, 320), (320, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf552, buf539, buf541, primals_343, mul_168, div_29, buf546, buf553, 1568, 320, grid=grid(1568), stream=stream0)
        del div_29
        del primals_343
        buf548 = reinterpret_tensor(buf543, (320, 13), (1, 320), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf539, buf541, mul_168, buf548, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_168
        buf549 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf548, buf549, 320, 13, grid=grid(320), stream=stream0)
        buf550 = reinterpret_tensor(buf548, (320, 13), (13, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf539, buf541, buf550, 4160, 121, grid=grid(4160), stream=stream0)
        buf551 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf550, buf551, 320, 13, grid=grid(320), stream=stream0)
        buf554 = reinterpret_tensor(buf501, (1568, 1280), (1280, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf553, permute_546, out=buf554)
        del permute_546
        buf555 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (320, 1568), (1, 320), 0), view_282, out=buf555)
        del view_282
        buf556 = reinterpret_tensor(buf550, (1, 320, 13), (4160, 1, 320), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf553, buf556, 4160, 121, grid=grid(4160), stream=stream0)
        buf557 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf556, buf557, 320, 13, grid=grid(320), stream=stream0)
        buf558 = reinterpret_tensor(buf554, (8, 196, 1280), (250880, 1280, 1), 0); del buf554  # reuse
        # Source Nodes: [x_307], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf558, addmm_88, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_88
        buf559 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (1568, 1280), (1280, 1), 0), permute_550, out=buf559)
        del permute_550
        buf560 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (1280, 1568), (1, 1280), 0), view_280, out=buf560)
        del view_280
        buf561 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf558, buf561, 16640, 121, grid=grid(16640), stream=stream0)
        buf562 = reinterpret_tensor(buf536, (1, 1280), (1280, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf561, buf562, 1280, 13, grid=grid(1280), stream=stream0)
        buf569 = buf552; del buf552  # reuse
        buf570 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf569, buf559, primals_337, mul_163, div_30, buf570, 1568, 320, grid=grid(1568), stream=stream0)
        del div_30
        del primals_337
        buf565 = reinterpret_tensor(buf556, (320, 13), (1, 320), 0); del buf556  # reuse
        buf567 = buf508; del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf559, mul_163, buf565, buf567, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_163
        buf566 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf565, buf566, 320, 13, grid=grid(320), stream=stream0)
        buf568 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf567, buf568, 320, 13, grid=grid(320), stream=stream0)
        buf571 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf570, permute_554, out=buf571)
        del permute_554
        buf572 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf570, (320, 1568), (1, 320), 0), view_278, out=buf572)
        del view_278
        buf573 = reinterpret_tensor(buf567, (1, 320, 13), (4160, 1, 320), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf570, buf573, 4160, 121, grid=grid(4160), stream=stream0)
        buf574 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf573, buf574, 320, 13, grid=grid(320), stream=stream0)
        buf575 = reinterpret_tensor(buf570, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_185, buf575, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_185
        buf576 = reinterpret_tensor(buf539, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_38, buf576, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_38
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf577 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf571, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf575, getitem_214, getitem_215, None, buf576, getitem_217, getitem_218, getitem_219, 0.0, [True, True, True, False])
        del buf571
        del buf575
        del getitem_214
        del getitem_215
        del getitem_217
        del getitem_218
        del getitem_219
        buf578 = buf577[0]
        buf579 = buf577[1]
        buf580 = buf577[2]
        del buf577
        buf581 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf579, buf580, buf581, 250880, grid=grid(250880), stream=stream0)
        buf582 = reinterpret_tensor(buf580, (392, 320), (320, 1), 0); del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (392, 640), (640, 1), 0), permute_560, out=buf582)
        del permute_560
        buf583 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf581, (640, 392), (1, 640), 0), view_274, out=buf583)
        del view_274
        buf584 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf581, buf584, 2560, 98, grid=grid(2560), stream=stream0)
        buf585 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf584, buf585, 640, 4, grid=grid(640), stream=stream0)
        buf592 = reinterpret_tensor(buf579, (8, 49, 320), (15680, 320, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf582, primals_331, mul_161, div_31, buf592, 392, 320, grid=grid(392), stream=stream0)
        del div_31
        del primals_331
        buf588 = buf531; del buf531  # reuse
        buf590 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf582, mul_161, buf588, buf590, 1280, 98, grid=grid(1280), stream=stream0)
        del buf582
        del mul_161
        buf589 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf588, buf589, 320, 4, grid=grid(320), stream=stream0)
        buf591 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf590, buf591, 320, 4, grid=grid(320), stream=stream0)
        buf593 = buf590; del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf592, buf593, 1280, 98, grid=grid(1280), stream=stream0)
        buf594 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf593, buf594, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf595 = aten.convolution_backward(reinterpret_tensor(buf592, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_272, primals_329, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf592
        del primals_329
        del view_272
        buf596 = buf595[0]
        buf597 = buf595[1]
        del buf595
        buf598 = reinterpret_tensor(buf576, (1568, 320), (320, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (1568, 320), (320, 1), 0), permute_567, out=buf598)
        del permute_567
        buf599 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (320, 1568), (1, 320), 0), view_269, out=buf599)
        del view_269
        buf600 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf578, buf600, 4160, 121, grid=grid(4160), stream=stream0)
        buf601 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf600, buf601, 320, 13, grid=grid(320), stream=stream0)
        buf602 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf596, buf598, primals_325, buf602, 4704, 107, grid=grid(4704), stream=stream0)
        buf603 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf602, buf603, 1568, 3, grid=grid(1568), stream=stream0)
        buf609 = buf569; del buf569  # reuse
        buf610 = reinterpret_tensor(buf578, (1568, 320), (320, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf609, buf596, buf598, primals_325, mul_159, div_32, buf603, buf610, 1568, 320, grid=grid(1568), stream=stream0)
        del div_32
        del primals_325
        buf605 = reinterpret_tensor(buf600, (320, 13), (1, 320), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf596, buf598, mul_159, buf605, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_159
        buf606 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf605, buf606, 320, 13, grid=grid(320), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (320, 13), (13, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf596, buf598, buf607, 4160, 121, grid=grid(4160), stream=stream0)
        buf608 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf607, buf608, 320, 13, grid=grid(320), stream=stream0)
        buf611 = reinterpret_tensor(buf558, (1568, 1280), (1280, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf610, permute_571, out=buf611)
        del permute_571
        buf612 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf610, (320, 1568), (1, 320), 0), view_267, out=buf612)
        del view_267
        buf613 = reinterpret_tensor(buf607, (1, 320, 13), (4160, 1, 320), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf610, buf613, 4160, 121, grid=grid(4160), stream=stream0)
        buf614 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf613, buf614, 320, 13, grid=grid(320), stream=stream0)
        buf615 = reinterpret_tensor(buf611, (8, 196, 1280), (250880, 1280, 1), 0); del buf611  # reuse
        # Source Nodes: [x_291], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf615, addmm_83, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_83
        buf616 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (1568, 1280), (1280, 1), 0), permute_575, out=buf616)
        del permute_575
        buf617 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf615, (1280, 1568), (1, 1280), 0), view_265, out=buf617)
        del view_265
        buf618 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf615, buf618, 16640, 121, grid=grid(16640), stream=stream0)
        buf619 = reinterpret_tensor(buf593, (1, 1280), (1280, 1), 0); del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf618, buf619, 1280, 13, grid=grid(1280), stream=stream0)
        buf626 = buf609; del buf609  # reuse
        buf627 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf626, buf616, primals_319, mul_154, div_33, buf627, 1568, 320, grid=grid(1568), stream=stream0)
        del div_33
        del primals_319
        buf622 = reinterpret_tensor(buf613, (320, 13), (1, 320), 0); del buf613  # reuse
        buf624 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf616, mul_154, buf622, buf624, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_154
        buf623 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf622, buf623, 320, 13, grid=grid(320), stream=stream0)
        buf625 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf624, buf625, 320, 13, grid=grid(320), stream=stream0)
        buf628 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf627, permute_579, out=buf628)
        del permute_579
        buf629 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf627, (320, 1568), (1, 320), 0), view_263, out=buf629)
        del view_263
        buf630 = reinterpret_tensor(buf624, (1, 320, 13), (4160, 1, 320), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf627, buf630, 4160, 121, grid=grid(4160), stream=stream0)
        buf631 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf630, buf631, 320, 13, grid=grid(320), stream=stream0)
        buf632 = reinterpret_tensor(buf627, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_175, buf632, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_175
        buf633 = reinterpret_tensor(buf596, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_39, buf633, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_39
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf634 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf628, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf632, getitem_202, getitem_203, None, buf633, getitem_205, getitem_206, getitem_207, 0.0, [True, True, True, False])
        del buf628
        del buf632
        del getitem_202
        del getitem_203
        del getitem_205
        del getitem_206
        del getitem_207
        buf635 = buf634[0]
        buf636 = buf634[1]
        buf637 = buf634[2]
        del buf634
        buf638 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf636, buf637, buf638, 250880, grid=grid(250880), stream=stream0)
        buf639 = reinterpret_tensor(buf637, (392, 320), (320, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (392, 640), (640, 1), 0), permute_585, out=buf639)
        del permute_585
        buf640 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf638, (640, 392), (1, 640), 0), view_259, out=buf640)
        del view_259
        buf641 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf638, buf641, 2560, 98, grid=grid(2560), stream=stream0)
        buf642 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf641, buf642, 640, 4, grid=grid(640), stream=stream0)
        buf649 = reinterpret_tensor(buf636, (8, 49, 320), (15680, 320, 1), 0); del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf639, primals_313, mul_152, div_34, buf649, 392, 320, grid=grid(392), stream=stream0)
        del div_34
        del primals_313
        buf645 = buf588; del buf588  # reuse
        buf647 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf639, mul_152, buf645, buf647, 1280, 98, grid=grid(1280), stream=stream0)
        del buf639
        del mul_152
        buf646 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf645, buf646, 320, 4, grid=grid(320), stream=stream0)
        buf648 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf647, buf648, 320, 4, grid=grid(320), stream=stream0)
        buf650 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf649, buf650, 1280, 98, grid=grid(1280), stream=stream0)
        buf651 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf650, buf651, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf652 = aten.convolution_backward(reinterpret_tensor(buf649, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_257, primals_311, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf649
        del primals_311
        del view_257
        buf653 = buf652[0]
        buf654 = buf652[1]
        del buf652
        buf655 = reinterpret_tensor(buf633, (1568, 320), (320, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (1568, 320), (320, 1), 0), permute_592, out=buf655)
        del permute_592
        buf656 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (320, 1568), (1, 320), 0), view_254, out=buf656)
        del view_254
        buf657 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf635, buf657, 4160, 121, grid=grid(4160), stream=stream0)
        buf658 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf657, buf658, 320, 13, grid=grid(320), stream=stream0)
        buf659 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf653, buf655, primals_307, buf659, 4704, 107, grid=grid(4704), stream=stream0)
        buf660 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf659, buf660, 1568, 3, grid=grid(1568), stream=stream0)
        buf666 = buf626; del buf626  # reuse
        buf667 = reinterpret_tensor(buf635, (1568, 320), (320, 1), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf666, buf653, buf655, primals_307, mul_150, div_35, buf660, buf667, 1568, 320, grid=grid(1568), stream=stream0)
        del div_35
        del primals_307
        buf662 = reinterpret_tensor(buf657, (320, 13), (1, 320), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf653, buf655, mul_150, buf662, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_150
        buf663 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf662, buf663, 320, 13, grid=grid(320), stream=stream0)
        buf664 = reinterpret_tensor(buf662, (320, 13), (13, 1), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf653, buf655, buf664, 4160, 121, grid=grid(4160), stream=stream0)
        buf665 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf664, buf665, 320, 13, grid=grid(320), stream=stream0)
        buf668 = reinterpret_tensor(buf615, (1568, 1280), (1280, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf667, permute_596, out=buf668)
        del permute_596
        buf669 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf667, (320, 1568), (1, 320), 0), view_252, out=buf669)
        del view_252
        buf670 = reinterpret_tensor(buf664, (1, 320, 13), (4160, 1, 320), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf667, buf670, 4160, 121, grid=grid(4160), stream=stream0)
        buf671 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf670, buf671, 320, 13, grid=grid(320), stream=stream0)
        buf672 = reinterpret_tensor(buf668, (8, 196, 1280), (250880, 1280, 1), 0); del buf668  # reuse
        # Source Nodes: [x_275], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf672, addmm_78, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_78
        buf673 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf672, (1568, 1280), (1280, 1), 0), permute_600, out=buf673)
        del permute_600
        buf674 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf672, (1280, 1568), (1, 1280), 0), view_250, out=buf674)
        del view_250
        buf675 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf672, buf675, 16640, 121, grid=grid(16640), stream=stream0)
        buf676 = reinterpret_tensor(buf650, (1, 1280), (1280, 1), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf675, buf676, 1280, 13, grid=grid(1280), stream=stream0)
        buf683 = buf666; del buf666  # reuse
        buf684 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf683, buf673, primals_301, mul_145, div_36, buf684, 1568, 320, grid=grid(1568), stream=stream0)
        del div_36
        del primals_301
        buf679 = reinterpret_tensor(buf670, (320, 13), (1, 320), 0); del buf670  # reuse
        buf681 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf673, mul_145, buf679, buf681, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_145
        buf680 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf679, buf680, 320, 13, grid=grid(320), stream=stream0)
        buf682 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf681, buf682, 320, 13, grid=grid(320), stream=stream0)
        buf685 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf684, permute_604, out=buf685)
        del permute_604
        buf686 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf684, (320, 1568), (1, 320), 0), view_248, out=buf686)
        del view_248
        buf687 = reinterpret_tensor(buf681, (1, 320, 13), (4160, 1, 320), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf684, buf687, 4160, 121, grid=grid(4160), stream=stream0)
        buf688 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf687, buf688, 320, 13, grid=grid(320), stream=stream0)
        buf689 = reinterpret_tensor(buf684, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_165, buf689, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_165
        buf690 = reinterpret_tensor(buf653, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_40, buf690, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_40
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf691 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf685, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf689, getitem_190, getitem_191, None, buf690, getitem_193, getitem_194, getitem_195, 0.0, [True, True, True, False])
        del buf685
        del buf689
        del getitem_190
        del getitem_191
        del getitem_193
        del getitem_194
        del getitem_195
        buf692 = buf691[0]
        buf693 = buf691[1]
        buf694 = buf691[2]
        del buf691
        buf695 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf693, buf694, buf695, 250880, grid=grid(250880), stream=stream0)
        buf696 = reinterpret_tensor(buf694, (392, 320), (320, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (392, 640), (640, 1), 0), permute_610, out=buf696)
        del permute_610
        buf697 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (640, 392), (1, 640), 0), view_244, out=buf697)
        del view_244
        buf698 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf695, buf698, 2560, 98, grid=grid(2560), stream=stream0)
        buf699 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf698, buf699, 640, 4, grid=grid(640), stream=stream0)
        buf706 = reinterpret_tensor(buf693, (8, 49, 320), (15680, 320, 1), 0); del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf696, primals_295, mul_143, div_37, buf706, 392, 320, grid=grid(392), stream=stream0)
        del div_37
        del primals_295
        buf702 = buf645; del buf645  # reuse
        buf704 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf696, mul_143, buf702, buf704, 1280, 98, grid=grid(1280), stream=stream0)
        del buf696
        del mul_143
        buf703 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf702, buf703, 320, 4, grid=grid(320), stream=stream0)
        buf705 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf704, buf705, 320, 4, grid=grid(320), stream=stream0)
        buf707 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf706, buf707, 1280, 98, grid=grid(1280), stream=stream0)
        buf708 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf707, buf708, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf709 = aten.convolution_backward(reinterpret_tensor(buf706, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_242, primals_293, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf706
        del primals_293
        del view_242
        buf710 = buf709[0]
        buf711 = buf709[1]
        del buf709
        buf712 = reinterpret_tensor(buf690, (1568, 320), (320, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf692, (1568, 320), (320, 1), 0), permute_617, out=buf712)
        del permute_617
        buf713 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf692, (320, 1568), (1, 320), 0), view_239, out=buf713)
        del view_239
        buf714 = buf687; del buf687  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf692, buf714, 4160, 121, grid=grid(4160), stream=stream0)
        buf715 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf714, buf715, 320, 13, grid=grid(320), stream=stream0)
        buf716 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf710, buf712, primals_289, buf716, 4704, 107, grid=grid(4704), stream=stream0)
        buf717 = buf660; del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf716, buf717, 1568, 3, grid=grid(1568), stream=stream0)
        buf723 = buf683; del buf683  # reuse
        buf724 = reinterpret_tensor(buf692, (1568, 320), (320, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf723, buf710, buf712, primals_289, mul_141, div_38, buf717, buf724, 1568, 320, grid=grid(1568), stream=stream0)
        del div_38
        del primals_289
        buf719 = reinterpret_tensor(buf714, (320, 13), (1, 320), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf710, buf712, mul_141, buf719, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_141
        buf720 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf719, buf720, 320, 13, grid=grid(320), stream=stream0)
        buf721 = reinterpret_tensor(buf719, (320, 13), (13, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf710, buf712, buf721, 4160, 121, grid=grid(4160), stream=stream0)
        buf722 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf721, buf722, 320, 13, grid=grid(320), stream=stream0)
        buf725 = reinterpret_tensor(buf672, (1568, 1280), (1280, 1), 0); del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf724, permute_621, out=buf725)
        del permute_621
        buf726 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (320, 1568), (1, 320), 0), view_237, out=buf726)
        del view_237
        buf727 = reinterpret_tensor(buf721, (1, 320, 13), (4160, 1, 320), 0); del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf724, buf727, 4160, 121, grid=grid(4160), stream=stream0)
        buf728 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf727, buf728, 320, 13, grid=grid(320), stream=stream0)
        buf729 = reinterpret_tensor(buf725, (8, 196, 1280), (250880, 1280, 1), 0); del buf725  # reuse
        # Source Nodes: [x_259], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf729, addmm_73, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_73
        buf730 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf729, (1568, 1280), (1280, 1), 0), permute_625, out=buf730)
        del permute_625
        buf731 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf729, (1280, 1568), (1, 1280), 0), view_235, out=buf731)
        del view_235
        buf732 = buf675; del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf729, buf732, 16640, 121, grid=grid(16640), stream=stream0)
        buf733 = reinterpret_tensor(buf707, (1, 1280), (1280, 1), 0); del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf732, buf733, 1280, 13, grid=grid(1280), stream=stream0)
        buf740 = buf723; del buf723  # reuse
        buf741 = buf712; del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf740, buf730, primals_283, mul_136, div_39, buf741, 1568, 320, grid=grid(1568), stream=stream0)
        del div_39
        del primals_283
        buf736 = reinterpret_tensor(buf727, (320, 13), (1, 320), 0); del buf727  # reuse
        buf738 = buf679; del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf730, mul_136, buf736, buf738, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_136
        buf737 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf736, buf737, 320, 13, grid=grid(320), stream=stream0)
        buf739 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf738, buf739, 320, 13, grid=grid(320), stream=stream0)
        buf742 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf741, permute_629, out=buf742)
        del permute_629
        buf743 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (320, 1568), (1, 320), 0), view_233, out=buf743)
        del view_233
        buf744 = reinterpret_tensor(buf738, (1, 320, 13), (4160, 1, 320), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf741, buf744, 4160, 121, grid=grid(4160), stream=stream0)
        buf745 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf744, buf745, 320, 13, grid=grid(320), stream=stream0)
        buf746 = reinterpret_tensor(buf741, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_155, buf746, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_155
        buf747 = reinterpret_tensor(buf710, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_41, buf747, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_41
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf748 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf742, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf746, getitem_178, getitem_179, None, buf747, getitem_181, getitem_182, getitem_183, 0.0, [True, True, True, False])
        del buf742
        del buf746
        del getitem_178
        del getitem_179
        del getitem_181
        del getitem_182
        del getitem_183
        buf749 = buf748[0]
        buf750 = buf748[1]
        buf751 = buf748[2]
        del buf748
        buf752 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf750, buf751, buf752, 250880, grid=grid(250880), stream=stream0)
        buf753 = reinterpret_tensor(buf751, (392, 320), (320, 1), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf752, (392, 640), (640, 1), 0), permute_635, out=buf753)
        del permute_635
        buf754 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf752, (640, 392), (1, 640), 0), view_229, out=buf754)
        del view_229
        buf755 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf752, buf755, 2560, 98, grid=grid(2560), stream=stream0)
        buf756 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf755, buf756, 640, 4, grid=grid(640), stream=stream0)
        buf763 = reinterpret_tensor(buf750, (8, 49, 320), (15680, 320, 1), 0); del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf753, primals_277, mul_134, div_40, buf763, 392, 320, grid=grid(392), stream=stream0)
        del div_40
        del primals_277
        buf759 = buf702; del buf702  # reuse
        buf761 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf753, mul_134, buf759, buf761, 1280, 98, grid=grid(1280), stream=stream0)
        del buf753
        del mul_134
        buf760 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf759, buf760, 320, 4, grid=grid(320), stream=stream0)
        buf762 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf761, buf762, 320, 4, grid=grid(320), stream=stream0)
        buf764 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf763, buf764, 1280, 98, grid=grid(1280), stream=stream0)
        buf765 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf764, buf765, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf766 = aten.convolution_backward(reinterpret_tensor(buf763, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_227, primals_275, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf763
        del primals_275
        del view_227
        buf767 = buf766[0]
        buf768 = buf766[1]
        del buf766
        buf769 = reinterpret_tensor(buf747, (1568, 320), (320, 1), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (1568, 320), (320, 1), 0), permute_642, out=buf769)
        del permute_642
        buf770 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (320, 1568), (1, 320), 0), view_224, out=buf770)
        del view_224
        buf771 = buf744; del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf749, buf771, 4160, 121, grid=grid(4160), stream=stream0)
        buf772 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf771, buf772, 320, 13, grid=grid(320), stream=stream0)
        buf773 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf767, buf769, primals_271, buf773, 4704, 107, grid=grid(4704), stream=stream0)
        buf774 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf773, buf774, 1568, 3, grid=grid(1568), stream=stream0)
        buf780 = buf740; del buf740  # reuse
        buf781 = reinterpret_tensor(buf749, (1568, 320), (320, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf780, buf767, buf769, primals_271, mul_132, div_41, buf774, buf781, 1568, 320, grid=grid(1568), stream=stream0)
        del div_41
        del primals_271
        buf776 = reinterpret_tensor(buf771, (320, 13), (1, 320), 0); del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf767, buf769, mul_132, buf776, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_132
        buf777 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf776, buf777, 320, 13, grid=grid(320), stream=stream0)
        buf778 = reinterpret_tensor(buf776, (320, 13), (13, 1), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf767, buf769, buf778, 4160, 121, grid=grid(4160), stream=stream0)
        buf779 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf778, buf779, 320, 13, grid=grid(320), stream=stream0)
        buf782 = reinterpret_tensor(buf729, (1568, 1280), (1280, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf781, permute_646, out=buf782)
        del permute_646
        buf783 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (320, 1568), (1, 320), 0), view_222, out=buf783)
        del view_222
        buf784 = reinterpret_tensor(buf778, (1, 320, 13), (4160, 1, 320), 0); del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf781, buf784, 4160, 121, grid=grid(4160), stream=stream0)
        buf785 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf784, buf785, 320, 13, grid=grid(320), stream=stream0)
        buf786 = reinterpret_tensor(buf782, (8, 196, 1280), (250880, 1280, 1), 0); del buf782  # reuse
        # Source Nodes: [x_243], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf786, addmm_68, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_68
        buf787 = buf781; del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (1568, 1280), (1280, 1), 0), permute_650, out=buf787)
        del permute_650
        buf788 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (1280, 1568), (1, 1280), 0), view_220, out=buf788)
        del view_220
        buf789 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf786, buf789, 16640, 121, grid=grid(16640), stream=stream0)
        buf790 = reinterpret_tensor(buf764, (1, 1280), (1280, 1), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf789, buf790, 1280, 13, grid=grid(1280), stream=stream0)
        buf797 = buf780; del buf780  # reuse
        buf798 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf797, buf787, primals_265, mul_127, div_42, buf798, 1568, 320, grid=grid(1568), stream=stream0)
        del div_42
        del primals_265
        buf793 = reinterpret_tensor(buf784, (320, 13), (1, 320), 0); del buf784  # reuse
        buf795 = buf736; del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf787, mul_127, buf793, buf795, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_127
        buf794 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf793, buf794, 320, 13, grid=grid(320), stream=stream0)
        buf796 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf795, buf796, 320, 13, grid=grid(320), stream=stream0)
        buf799 = buf787; del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf798, permute_654, out=buf799)
        del permute_654
        buf800 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (320, 1568), (1, 320), 0), view_218, out=buf800)
        del view_218
        buf801 = reinterpret_tensor(buf795, (1, 320, 13), (4160, 1, 320), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf798, buf801, 4160, 121, grid=grid(4160), stream=stream0)
        buf802 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf801, buf802, 320, 13, grid=grid(320), stream=stream0)
        buf803 = reinterpret_tensor(buf798, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_145, buf803, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_145
        buf804 = reinterpret_tensor(buf767, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_42, buf804, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_42
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf805 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf799, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf803, getitem_166, getitem_167, None, buf804, getitem_169, getitem_170, getitem_171, 0.0, [True, True, True, False])
        del buf799
        del buf803
        del getitem_166
        del getitem_167
        del getitem_169
        del getitem_170
        del getitem_171
        buf806 = buf805[0]
        buf807 = buf805[1]
        buf808 = buf805[2]
        del buf805
        buf809 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf807, buf808, buf809, 250880, grid=grid(250880), stream=stream0)
        buf810 = reinterpret_tensor(buf808, (392, 320), (320, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (392, 640), (640, 1), 0), permute_660, out=buf810)
        del permute_660
        buf811 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (640, 392), (1, 640), 0), view_214, out=buf811)
        del view_214
        buf812 = buf755; del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf809, buf812, 2560, 98, grid=grid(2560), stream=stream0)
        buf813 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf812, buf813, 640, 4, grid=grid(640), stream=stream0)
        buf820 = reinterpret_tensor(buf807, (8, 49, 320), (15680, 320, 1), 0); del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf810, primals_259, mul_125, div_43, buf820, 392, 320, grid=grid(392), stream=stream0)
        del div_43
        del primals_259
        buf816 = buf759; del buf759  # reuse
        buf818 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf810, mul_125, buf816, buf818, 1280, 98, grid=grid(1280), stream=stream0)
        del buf810
        del mul_125
        buf817 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf816, buf817, 320, 4, grid=grid(320), stream=stream0)
        buf819 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf818, buf819, 320, 4, grid=grid(320), stream=stream0)
        buf821 = buf818; del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf820, buf821, 1280, 98, grid=grid(1280), stream=stream0)
        buf822 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf821, buf822, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf823 = aten.convolution_backward(reinterpret_tensor(buf820, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_212, primals_257, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf820
        del primals_257
        del view_212
        buf824 = buf823[0]
        buf825 = buf823[1]
        del buf823
        buf826 = reinterpret_tensor(buf804, (1568, 320), (320, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (1568, 320), (320, 1), 0), permute_667, out=buf826)
        del permute_667
        buf827 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (320, 1568), (1, 320), 0), view_209, out=buf827)
        del view_209
        buf828 = buf801; del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf806, buf828, 4160, 121, grid=grid(4160), stream=stream0)
        buf829 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf828, buf829, 320, 13, grid=grid(320), stream=stream0)
        buf830 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf824, buf826, primals_253, buf830, 4704, 107, grid=grid(4704), stream=stream0)
        buf831 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf830, buf831, 1568, 3, grid=grid(1568), stream=stream0)
        buf837 = buf797; del buf797  # reuse
        buf838 = reinterpret_tensor(buf806, (1568, 320), (320, 1), 0); del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf837, buf824, buf826, primals_253, mul_123, div_44, buf831, buf838, 1568, 320, grid=grid(1568), stream=stream0)
        del div_44
        del primals_253
        buf833 = reinterpret_tensor(buf828, (320, 13), (1, 320), 0); del buf828  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf824, buf826, mul_123, buf833, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_123
        buf834 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf833, buf834, 320, 13, grid=grid(320), stream=stream0)
        buf835 = reinterpret_tensor(buf833, (320, 13), (13, 1), 0); del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf824, buf826, buf835, 4160, 121, grid=grid(4160), stream=stream0)
        buf836 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf835, buf836, 320, 13, grid=grid(320), stream=stream0)
        buf839 = reinterpret_tensor(buf786, (1568, 1280), (1280, 1), 0); del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf838, permute_671, out=buf839)
        del permute_671
        buf840 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf838, (320, 1568), (1, 320), 0), view_207, out=buf840)
        del view_207
        buf841 = reinterpret_tensor(buf835, (1, 320, 13), (4160, 1, 320), 0); del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf838, buf841, 4160, 121, grid=grid(4160), stream=stream0)
        buf842 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf841, buf842, 320, 13, grid=grid(320), stream=stream0)
        buf843 = reinterpret_tensor(buf839, (8, 196, 1280), (250880, 1280, 1), 0); del buf839  # reuse
        # Source Nodes: [x_227], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf843, addmm_63, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_63
        buf844 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (1568, 1280), (1280, 1), 0), permute_675, out=buf844)
        del permute_675
        buf845 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf843, (1280, 1568), (1, 1280), 0), view_205, out=buf845)
        del view_205
        buf846 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf843, buf846, 16640, 121, grid=grid(16640), stream=stream0)
        buf847 = reinterpret_tensor(buf821, (1, 1280), (1280, 1), 0); del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf846, buf847, 1280, 13, grid=grid(1280), stream=stream0)
        buf854 = buf837; del buf837  # reuse
        buf855 = buf826; del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf854, buf844, primals_247, mul_118, div_45, buf855, 1568, 320, grid=grid(1568), stream=stream0)
        del div_45
        del primals_247
        buf850 = reinterpret_tensor(buf841, (320, 13), (1, 320), 0); del buf841  # reuse
        buf852 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf844, mul_118, buf850, buf852, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_118
        buf851 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf850, buf851, 320, 13, grid=grid(320), stream=stream0)
        buf853 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf852, buf853, 320, 13, grid=grid(320), stream=stream0)
        buf856 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf855, permute_679, out=buf856)
        del permute_679
        buf857 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf855, (320, 1568), (1, 320), 0), view_203, out=buf857)
        del view_203
        buf858 = reinterpret_tensor(buf852, (1, 320, 13), (4160, 1, 320), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf855, buf858, 4160, 121, grid=grid(4160), stream=stream0)
        buf859 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf858, buf859, 320, 13, grid=grid(320), stream=stream0)
        buf860 = reinterpret_tensor(buf855, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_135, buf860, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_135
        buf861 = reinterpret_tensor(buf824, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_43, buf861, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_43
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf862 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf856, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf860, getitem_154, getitem_155, None, buf861, getitem_157, getitem_158, getitem_159, 0.0, [True, True, True, False])
        del buf856
        del buf860
        del getitem_154
        del getitem_155
        del getitem_157
        del getitem_158
        del getitem_159
        buf863 = buf862[0]
        buf864 = buf862[1]
        buf865 = buf862[2]
        del buf862
        buf866 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf864, buf865, buf866, 250880, grid=grid(250880), stream=stream0)
        buf867 = reinterpret_tensor(buf865, (392, 320), (320, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf866, (392, 640), (640, 1), 0), permute_685, out=buf867)
        del permute_685
        buf868 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf866, (640, 392), (1, 640), 0), view_199, out=buf868)
        del view_199
        buf869 = buf812; del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf866, buf869, 2560, 98, grid=grid(2560), stream=stream0)
        buf870 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf869, buf870, 640, 4, grid=grid(640), stream=stream0)
        buf877 = reinterpret_tensor(buf864, (8, 49, 320), (15680, 320, 1), 0); del buf864  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf867, primals_241, mul_116, div_46, buf877, 392, 320, grid=grid(392), stream=stream0)
        del div_46
        del primals_241
        buf873 = buf816; del buf816  # reuse
        buf875 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf867, mul_116, buf873, buf875, 1280, 98, grid=grid(1280), stream=stream0)
        del buf867
        del mul_116
        buf874 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf873, buf874, 320, 4, grid=grid(320), stream=stream0)
        buf876 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf875, buf876, 320, 4, grid=grid(320), stream=stream0)
        buf878 = buf875; del buf875  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf877, buf878, 1280, 98, grid=grid(1280), stream=stream0)
        buf879 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf878, buf879, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf880 = aten.convolution_backward(reinterpret_tensor(buf877, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_197, primals_239, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf877
        del primals_239
        del view_197
        buf881 = buf880[0]
        buf882 = buf880[1]
        del buf880
        buf883 = reinterpret_tensor(buf861, (1568, 320), (320, 1), 0); del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (1568, 320), (320, 1), 0), permute_692, out=buf883)
        del permute_692
        buf884 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf863, (320, 1568), (1, 320), 0), view_194, out=buf884)
        del view_194
        buf885 = buf858; del buf858  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf863, buf885, 4160, 121, grid=grid(4160), stream=stream0)
        buf886 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf885, buf886, 320, 13, grid=grid(320), stream=stream0)
        buf887 = buf830; del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf881, buf883, primals_235, buf887, 4704, 107, grid=grid(4704), stream=stream0)
        buf888 = buf831; del buf831  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf887, buf888, 1568, 3, grid=grid(1568), stream=stream0)
        buf894 = buf854; del buf854  # reuse
        buf895 = reinterpret_tensor(buf863, (1568, 320), (320, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf894, buf881, buf883, primals_235, mul_114, div_47, buf888, buf895, 1568, 320, grid=grid(1568), stream=stream0)
        del div_47
        del primals_235
        buf890 = reinterpret_tensor(buf885, (320, 13), (1, 320), 0); del buf885  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf881, buf883, mul_114, buf890, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_114
        buf891 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf890, buf891, 320, 13, grid=grid(320), stream=stream0)
        buf892 = reinterpret_tensor(buf890, (320, 13), (13, 1), 0); del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf881, buf883, buf892, 4160, 121, grid=grid(4160), stream=stream0)
        buf893 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf892, buf893, 320, 13, grid=grid(320), stream=stream0)
        buf896 = reinterpret_tensor(buf843, (1568, 1280), (1280, 1), 0); del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf895, permute_696, out=buf896)
        del permute_696
        buf897 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf895, (320, 1568), (1, 320), 0), view_192, out=buf897)
        del view_192
        buf898 = reinterpret_tensor(buf892, (1, 320, 13), (4160, 1, 320), 0); del buf892  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf895, buf898, 4160, 121, grid=grid(4160), stream=stream0)
        buf899 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf898, buf899, 320, 13, grid=grid(320), stream=stream0)
        buf900 = reinterpret_tensor(buf896, (8, 196, 1280), (250880, 1280, 1), 0); del buf896  # reuse
        # Source Nodes: [x_211], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf900, addmm_58, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_58
        buf901 = buf895; del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf900, (1568, 1280), (1280, 1), 0), permute_700, out=buf901)
        del permute_700
        buf902 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf900, (1280, 1568), (1, 1280), 0), view_190, out=buf902)
        del view_190
        buf903 = buf846; del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf900, buf903, 16640, 121, grid=grid(16640), stream=stream0)
        buf904 = reinterpret_tensor(buf878, (1, 1280), (1280, 1), 0); del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf903, buf904, 1280, 13, grid=grid(1280), stream=stream0)
        buf911 = buf894; del buf894  # reuse
        buf912 = buf883; del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf911, buf901, primals_229, mul_109, div_48, buf912, 1568, 320, grid=grid(1568), stream=stream0)
        del div_48
        del primals_229
        buf907 = reinterpret_tensor(buf898, (320, 13), (1, 320), 0); del buf898  # reuse
        buf909 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf901, mul_109, buf907, buf909, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_109
        buf908 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf907, buf908, 320, 13, grid=grid(320), stream=stream0)
        buf910 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf909, buf910, 320, 13, grid=grid(320), stream=stream0)
        buf913 = buf901; del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf912, permute_704, out=buf913)
        del permute_704
        buf914 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf912, (320, 1568), (1, 320), 0), view_188, out=buf914)
        del view_188
        buf915 = reinterpret_tensor(buf909, (1, 320, 13), (4160, 1, 320), 0); del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf912, buf915, 4160, 121, grid=grid(4160), stream=stream0)
        buf916 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf915, buf916, 320, 13, grid=grid(320), stream=stream0)
        buf917 = reinterpret_tensor(buf912, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf912  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_125, buf917, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_125
        buf918 = reinterpret_tensor(buf881, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf881  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_44, buf918, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_44
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf919 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf913, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf917, getitem_142, getitem_143, None, buf918, getitem_145, getitem_146, getitem_147, 0.0, [True, True, True, False])
        del buf913
        del buf917
        del getitem_142
        del getitem_143
        del getitem_145
        del getitem_146
        del getitem_147
        buf920 = buf919[0]
        buf921 = buf919[1]
        buf922 = buf919[2]
        del buf919
        buf923 = buf866; del buf866  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf921, buf922, buf923, 250880, grid=grid(250880), stream=stream0)
        buf924 = reinterpret_tensor(buf922, (392, 320), (320, 1), 0); del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf923, (392, 640), (640, 1), 0), permute_710, out=buf924)
        del permute_710
        buf925 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf923, (640, 392), (1, 640), 0), view_184, out=buf925)
        del view_184
        buf926 = buf869; del buf869  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf923, buf926, 2560, 98, grid=grid(2560), stream=stream0)
        buf927 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf926, buf927, 640, 4, grid=grid(640), stream=stream0)
        buf934 = reinterpret_tensor(buf921, (8, 49, 320), (15680, 320, 1), 0); del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf924, primals_223, mul_107, div_49, buf934, 392, 320, grid=grid(392), stream=stream0)
        del div_49
        del primals_223
        buf930 = buf873; del buf873  # reuse
        buf932 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf924, mul_107, buf930, buf932, 1280, 98, grid=grid(1280), stream=stream0)
        del buf924
        del mul_107
        buf931 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf930, buf931, 320, 4, grid=grid(320), stream=stream0)
        buf933 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf932, buf933, 320, 4, grid=grid(320), stream=stream0)
        buf935 = buf932; del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf934, buf935, 1280, 98, grid=grid(1280), stream=stream0)
        buf936 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf935, buf936, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf937 = aten.convolution_backward(reinterpret_tensor(buf934, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_182, primals_221, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf934
        del primals_221
        del view_182
        buf938 = buf937[0]
        buf939 = buf937[1]
        del buf937
        buf940 = reinterpret_tensor(buf918, (1568, 320), (320, 1), 0); del buf918  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (1568, 320), (320, 1), 0), permute_717, out=buf940)
        del permute_717
        buf941 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (320, 1568), (1, 320), 0), view_179, out=buf941)
        del view_179
        buf942 = buf915; del buf915  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf920, buf942, 4160, 121, grid=grid(4160), stream=stream0)
        buf943 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf942, buf943, 320, 13, grid=grid(320), stream=stream0)
        buf944 = buf887; del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf938, buf940, primals_217, buf944, 4704, 107, grid=grid(4704), stream=stream0)
        buf945 = buf888; del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf944, buf945, 1568, 3, grid=grid(1568), stream=stream0)
        buf951 = buf911; del buf911  # reuse
        buf952 = reinterpret_tensor(buf920, (1568, 320), (320, 1), 0); del buf920  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf951, buf938, buf940, primals_217, mul_105, div_50, buf945, buf952, 1568, 320, grid=grid(1568), stream=stream0)
        del div_50
        del primals_217
        buf947 = reinterpret_tensor(buf942, (320, 13), (1, 320), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf938, buf940, mul_105, buf947, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_105
        buf948 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf947, buf948, 320, 13, grid=grid(320), stream=stream0)
        buf949 = reinterpret_tensor(buf947, (320, 13), (13, 1), 0); del buf947  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf938, buf940, buf949, 4160, 121, grid=grid(4160), stream=stream0)
        buf950 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf949, buf950, 320, 13, grid=grid(320), stream=stream0)
        buf953 = reinterpret_tensor(buf900, (1568, 1280), (1280, 1), 0); del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf952, permute_721, out=buf953)
        del permute_721
        buf954 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf952, (320, 1568), (1, 320), 0), view_177, out=buf954)
        del view_177
        buf955 = reinterpret_tensor(buf949, (1, 320, 13), (4160, 1, 320), 0); del buf949  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf952, buf955, 4160, 121, grid=grid(4160), stream=stream0)
        buf956 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf955, buf956, 320, 13, grid=grid(320), stream=stream0)
        buf957 = reinterpret_tensor(buf953, (8, 196, 1280), (250880, 1280, 1), 0); del buf953  # reuse
        # Source Nodes: [x_195], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf957, addmm_53, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_53
        buf958 = buf952; del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (1568, 1280), (1280, 1), 0), permute_725, out=buf958)
        del permute_725
        buf959 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (1280, 1568), (1, 1280), 0), view_175, out=buf959)
        del view_175
        buf960 = buf903; del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf957, buf960, 16640, 121, grid=grid(16640), stream=stream0)
        buf961 = reinterpret_tensor(buf935, (1, 1280), (1280, 1), 0); del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf960, buf961, 1280, 13, grid=grid(1280), stream=stream0)
        buf968 = buf951; del buf951  # reuse
        buf969 = buf940; del buf940  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf968, buf958, primals_211, mul_100, div_51, buf969, 1568, 320, grid=grid(1568), stream=stream0)
        del div_51
        del primals_211
        buf964 = reinterpret_tensor(buf955, (320, 13), (1, 320), 0); del buf955  # reuse
        buf966 = buf907; del buf907  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf958, mul_100, buf964, buf966, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_100
        buf965 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf964, buf965, 320, 13, grid=grid(320), stream=stream0)
        buf967 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf966, buf967, 320, 13, grid=grid(320), stream=stream0)
        buf970 = buf958; del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf969, permute_729, out=buf970)
        del permute_729
        buf971 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf969, (320, 1568), (1, 320), 0), view_173, out=buf971)
        del view_173
        buf972 = reinterpret_tensor(buf966, (1, 320, 13), (4160, 1, 320), 0); del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf969, buf972, 4160, 121, grid=grid(4160), stream=stream0)
        buf973 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf972, buf973, 320, 13, grid=grid(320), stream=stream0)
        buf974 = reinterpret_tensor(buf969, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf969  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_115, buf974, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_115
        buf975 = reinterpret_tensor(buf938, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf938  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_45, buf975, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_45
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf976 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf970, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf974, getitem_130, getitem_131, None, buf975, getitem_133, getitem_134, getitem_135, 0.0, [True, True, True, False])
        del buf970
        del buf974
        del getitem_130
        del getitem_131
        del getitem_133
        del getitem_134
        del getitem_135
        buf977 = buf976[0]
        buf978 = buf976[1]
        buf979 = buf976[2]
        del buf976
        buf980 = buf923; del buf923  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf978, buf979, buf980, 250880, grid=grid(250880), stream=stream0)
        buf981 = reinterpret_tensor(buf979, (392, 320), (320, 1), 0); del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (392, 640), (640, 1), 0), permute_735, out=buf981)
        del permute_735
        buf982 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (640, 392), (1, 640), 0), view_169, out=buf982)
        del view_169
        buf983 = buf926; del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf980, buf983, 2560, 98, grid=grid(2560), stream=stream0)
        buf984 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf983, buf984, 640, 4, grid=grid(640), stream=stream0)
        buf991 = reinterpret_tensor(buf978, (8, 49, 320), (15680, 320, 1), 0); del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf981, primals_205, mul_98, div_52, buf991, 392, 320, grid=grid(392), stream=stream0)
        del div_52
        del primals_205
        buf987 = buf930; del buf930  # reuse
        buf989 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf981, mul_98, buf987, buf989, 1280, 98, grid=grid(1280), stream=stream0)
        del buf981
        del mul_98
        buf988 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf987, buf988, 320, 4, grid=grid(320), stream=stream0)
        buf990 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf989, buf990, 320, 4, grid=grid(320), stream=stream0)
        buf992 = buf989; del buf989  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf991, buf992, 1280, 98, grid=grid(1280), stream=stream0)
        buf993 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf992, buf993, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf994 = aten.convolution_backward(reinterpret_tensor(buf991, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_167, primals_203, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf991
        del primals_203
        del view_167
        buf995 = buf994[0]
        buf996 = buf994[1]
        del buf994
        buf997 = reinterpret_tensor(buf975, (1568, 320), (320, 1), 0); del buf975  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf977, (1568, 320), (320, 1), 0), permute_742, out=buf997)
        del permute_742
        buf998 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf977, (320, 1568), (1, 320), 0), view_164, out=buf998)
        del view_164
        buf999 = buf972; del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf977, buf999, 4160, 121, grid=grid(4160), stream=stream0)
        buf1000 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf999, buf1000, 320, 13, grid=grid(320), stream=stream0)
        buf1001 = buf944; del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf995, buf997, primals_199, buf1001, 4704, 107, grid=grid(4704), stream=stream0)
        buf1002 = buf945; del buf945  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf1001, buf1002, 1568, 3, grid=grid(1568), stream=stream0)
        buf1008 = buf968; del buf968  # reuse
        buf1009 = reinterpret_tensor(buf977, (1568, 320), (320, 1), 0); del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf1008, buf995, buf997, primals_199, mul_96, div_53, buf1002, buf1009, 1568, 320, grid=grid(1568), stream=stream0)
        del div_53
        del primals_199
        buf1004 = reinterpret_tensor(buf999, (320, 13), (1, 320), 0); del buf999  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf995, buf997, mul_96, buf1004, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_96
        buf1005 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1004, buf1005, 320, 13, grid=grid(320), stream=stream0)
        buf1006 = reinterpret_tensor(buf1004, (320, 13), (13, 1), 0); del buf1004  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf995, buf997, buf1006, 4160, 121, grid=grid(4160), stream=stream0)
        buf1007 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf1006, buf1007, 320, 13, grid=grid(320), stream=stream0)
        buf1010 = reinterpret_tensor(buf957, (1568, 1280), (1280, 1), 0); del buf957  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1009, permute_746, out=buf1010)
        del permute_746
        buf1011 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1009, (320, 1568), (1, 320), 0), view_162, out=buf1011)
        del view_162
        buf1012 = reinterpret_tensor(buf1006, (1, 320, 13), (4160, 1, 320), 0); del buf1006  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1009, buf1012, 4160, 121, grid=grid(4160), stream=stream0)
        buf1013 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1012, buf1013, 320, 13, grid=grid(320), stream=stream0)
        buf1014 = reinterpret_tensor(buf1010, (8, 196, 1280), (250880, 1280, 1), 0); del buf1010  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1014, addmm_48, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_48
        buf1015 = buf1009; del buf1009  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1014, (1568, 1280), (1280, 1), 0), permute_750, out=buf1015)
        del permute_750
        buf1016 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1014, (1280, 1568), (1, 1280), 0), view_160, out=buf1016)
        del view_160
        buf1017 = buf960; del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1014, buf1017, 16640, 121, grid=grid(16640), stream=stream0)
        buf1018 = reinterpret_tensor(buf992, (1, 1280), (1280, 1), 0); del buf992  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1017, buf1018, 1280, 13, grid=grid(1280), stream=stream0)
        buf1025 = buf1008; del buf1008  # reuse
        buf1026 = buf997; del buf997  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf1025, buf1015, primals_193, mul_91, div_54, buf1026, 1568, 320, grid=grid(1568), stream=stream0)
        del div_54
        del primals_193
        buf1021 = reinterpret_tensor(buf1012, (320, 13), (1, 320), 0); del buf1012  # reuse
        buf1023 = buf964; del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1015, mul_91, buf1021, buf1023, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_91
        buf1022 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1021, buf1022, 320, 13, grid=grid(320), stream=stream0)
        buf1024 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1023, buf1024, 320, 13, grid=grid(320), stream=stream0)
        buf1027 = buf1015; del buf1015  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1026, permute_754, out=buf1027)
        del permute_754
        buf1028 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1026, (320, 1568), (1, 320), 0), view_158, out=buf1028)
        del view_158
        buf1029 = reinterpret_tensor(buf1023, (1, 320, 13), (4160, 1, 320), 0); del buf1023  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1026, buf1029, 4160, 121, grid=grid(4160), stream=stream0)
        buf1030 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1029, buf1030, 320, 13, grid=grid(320), stream=stream0)
        buf1031 = reinterpret_tensor(buf1026, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf1026  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_105, buf1031, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_105
        buf1032 = reinterpret_tensor(buf995, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf995  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_46, buf1032, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_46
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1033 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1027, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf1031, getitem_118, getitem_119, None, buf1032, getitem_121, getitem_122, getitem_123, 0.0, [True, True, True, False])
        del buf1027
        del buf1031
        del getitem_118
        del getitem_119
        del getitem_121
        del getitem_122
        del getitem_123
        buf1034 = buf1033[0]
        buf1035 = buf1033[1]
        buf1036 = buf1033[2]
        del buf1033
        buf1037 = buf980; del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf1035, buf1036, buf1037, 250880, grid=grid(250880), stream=stream0)
        buf1038 = reinterpret_tensor(buf1036, (392, 320), (320, 1), 0); del buf1036  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (392, 640), (640, 1), 0), permute_760, out=buf1038)
        del permute_760
        buf1039 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1037, (640, 392), (1, 640), 0), view_154, out=buf1039)
        del view_154
        buf1040 = buf983; del buf983  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf1037, buf1040, 2560, 98, grid=grid(2560), stream=stream0)
        buf1041 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf1040, buf1041, 640, 4, grid=grid(640), stream=stream0)
        buf1048 = reinterpret_tensor(buf1035, (8, 49, 320), (15680, 320, 1), 0); del buf1035  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf1038, primals_187, mul_89, div_55, buf1048, 392, 320, grid=grid(392), stream=stream0)
        del div_55
        del primals_187
        buf1044 = buf987; del buf987  # reuse
        buf1046 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf1038, mul_89, buf1044, buf1046, 1280, 98, grid=grid(1280), stream=stream0)
        del buf1038
        del mul_89
        buf1045 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1044, buf1045, 320, 4, grid=grid(320), stream=stream0)
        buf1047 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1046, buf1047, 320, 4, grid=grid(320), stream=stream0)
        buf1049 = buf1046; del buf1046  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf1048, buf1049, 1280, 98, grid=grid(1280), stream=stream0)
        buf1050 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1049, buf1050, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1051 = aten.convolution_backward(reinterpret_tensor(buf1048, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_152, primals_185, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1048
        del primals_185
        del view_152
        buf1052 = buf1051[0]
        buf1053 = buf1051[1]
        del buf1051
        buf1054 = reinterpret_tensor(buf1032, (1568, 320), (320, 1), 0); del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1034, (1568, 320), (320, 1), 0), permute_767, out=buf1054)
        del permute_767
        buf1055 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1034, (320, 1568), (1, 320), 0), view_149, out=buf1055)
        del view_149
        buf1056 = buf1029; del buf1029  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1034, buf1056, 4160, 121, grid=grid(4160), stream=stream0)
        buf1057 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1056, buf1057, 320, 13, grid=grid(320), stream=stream0)
        buf1058 = buf1001; del buf1001  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf1052, buf1054, primals_181, buf1058, 4704, 107, grid=grid(4704), stream=stream0)
        buf1059 = buf1002; del buf1002  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf1058, buf1059, 1568, 3, grid=grid(1568), stream=stream0)
        buf1065 = buf1025; del buf1025  # reuse
        buf1066 = reinterpret_tensor(buf1034, (1568, 320), (320, 1), 0); del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_40.run(buf1065, buf1052, buf1054, primals_181, mul_87, div_56, buf1059, buf1066, 1568, 320, grid=grid(1568), stream=stream0)
        del div_56
        del primals_181
        buf1061 = reinterpret_tensor(buf1056, (320, 13), (1, 320), 0); del buf1056  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf1052, buf1054, mul_87, buf1061, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_87
        buf1062 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1061, buf1062, 320, 13, grid=grid(320), stream=stream0)
        buf1063 = reinterpret_tensor(buf1061, (320, 13), (13, 1), 0); del buf1061  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf1052, buf1054, buf1063, 4160, 121, grid=grid(4160), stream=stream0)
        buf1064 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf1063, buf1064, 320, 13, grid=grid(320), stream=stream0)
        buf1067 = reinterpret_tensor(buf1014, (1568, 1280), (1280, 1), 0); del buf1014  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1066, permute_771, out=buf1067)
        del permute_771
        buf1068 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1066, (320, 1568), (1, 320), 0), view_147, out=buf1068)
        del view_147
        buf1069 = reinterpret_tensor(buf1063, (1, 320, 13), (4160, 1, 320), 0); del buf1063  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1066, buf1069, 4160, 121, grid=grid(4160), stream=stream0)
        buf1070 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1069, buf1070, 320, 13, grid=grid(320), stream=stream0)
        buf1071 = reinterpret_tensor(buf1067, (8, 196, 1280), (250880, 1280, 1), 0); del buf1067  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1071, addmm_43, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_43
        buf1072 = buf1066; del buf1066  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1071, (1568, 1280), (1280, 1), 0), permute_775, out=buf1072)
        del permute_775
        buf1073 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1071, (1280, 1568), (1, 1280), 0), view_145, out=buf1073)
        del view_145
        buf1074 = buf1017; del buf1017  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1071, buf1074, 16640, 121, grid=grid(16640), stream=stream0)
        buf1075 = reinterpret_tensor(buf1049, (1, 1280), (1280, 1), 0); del buf1049  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1074, buf1075, 1280, 13, grid=grid(1280), stream=stream0)
        buf1082 = buf1065; del buf1065  # reuse
        buf1083 = buf1054; del buf1054  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_44.run(buf1082, buf1072, primals_175, mul_82, div_57, buf1083, 1568, 320, grid=grid(1568), stream=stream0)
        del div_57
        del primals_175
        buf1078 = reinterpret_tensor(buf1069, (320, 13), (1, 320), 0); del buf1069  # reuse
        buf1080 = buf1021; del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1072, mul_82, buf1078, buf1080, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_82
        buf1079 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1078, buf1079, 320, 13, grid=grid(320), stream=stream0)
        buf1081 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1080, buf1081, 320, 13, grid=grid(320), stream=stream0)
        buf1084 = buf1072; del buf1072  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1083, permute_779, out=buf1084)
        del permute_779
        buf1085 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1083, (320, 1568), (1, 320), 0), view_143, out=buf1085)
        del view_143
        buf1086 = reinterpret_tensor(buf1080, (1, 320, 13), (4160, 1, 320), 0); del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1083, buf1086, 4160, 121, grid=grid(4160), stream=stream0)
        buf1087 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1086, buf1087, 320, 13, grid=grid(320), stream=stream0)
        buf1088 = reinterpret_tensor(buf1083, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf1083  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_95, buf1088, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_95
        buf1089 = reinterpret_tensor(buf1052, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf1052  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_47, buf1089, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_47
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1090 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1084, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf1088, getitem_106, getitem_107, None, buf1089, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False])
        del buf1084
        del buf1088
        del getitem_106
        del getitem_107
        del getitem_109
        del getitem_110
        del getitem_111
        buf1091 = buf1090[0]
        buf1092 = buf1090[1]
        buf1093 = buf1090[2]
        del buf1090
        buf1094 = buf1037; del buf1037  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf1092, buf1093, buf1094, 250880, grid=grid(250880), stream=stream0)
        buf1095 = reinterpret_tensor(buf1093, (392, 320), (320, 1), 0); del buf1093  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (392, 640), (640, 1), 0), permute_785, out=buf1095)
        del permute_785
        buf1096 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (640, 392), (1, 640), 0), view_139, out=buf1096)
        del view_139
        buf1097 = buf1040; del buf1040  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf1094, buf1097, 2560, 98, grid=grid(2560), stream=stream0)
        buf1098 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf1097, buf1098, 640, 4, grid=grid(640), stream=stream0)
        buf1105 = reinterpret_tensor(buf1092, (8, 49, 320), (15680, 320, 1), 0); del buf1092  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf1095, primals_169, mul_80, div_58, buf1105, 392, 320, grid=grid(392), stream=stream0)
        del div_58
        del primals_169
        buf1101 = buf1044; del buf1044  # reuse
        buf1103 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf1095, mul_80, buf1101, buf1103, 1280, 98, grid=grid(1280), stream=stream0)
        del buf1095
        del mul_80
        buf1102 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1101, buf1102, 320, 4, grid=grid(320), stream=stream0)
        buf1104 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1103, buf1104, 320, 4, grid=grid(320), stream=stream0)
        buf1106 = buf1103; del buf1103  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf1105, buf1106, 1280, 98, grid=grid(1280), stream=stream0)
        buf1107 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1106, buf1107, 320, 4, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1108 = aten.convolution_backward(reinterpret_tensor(buf1105, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_137, primals_167, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1105
        del primals_167
        del view_137
        buf1109 = buf1108[0]
        buf1110 = buf1108[1]
        del buf1108
        buf1111 = reinterpret_tensor(buf1089, (1568, 320), (320, 1), 0); del buf1089  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1091, (1568, 320), (320, 1), 0), permute_792, out=buf1111)
        del permute_792
        buf1112 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1091, (320, 1568), (1, 320), 0), view_134, out=buf1112)
        del view_134
        buf1113 = buf1086; del buf1086  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1091, buf1113, 4160, 121, grid=grid(4160), stream=stream0)
        buf1114 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1113, buf1114, 320, 13, grid=grid(320), stream=stream0)
        buf1115 = buf1058; del buf1058  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf1109, buf1111, primals_163, buf1115, 4704, 107, grid=grid(4704), stream=stream0)
        buf1116 = buf1059; del buf1059  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf1115, buf1116, 1568, 3, grid=grid(1568), stream=stream0)
        buf1122 = buf1082; del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_45.run(buf1122, buf1109, buf1111, primals_163, mul_78, div_59, buf1116, 1568, 320, grid=grid(1568), stream=stream0)
        del div_59
        del primals_163
        buf1118 = reinterpret_tensor(buf1113, (320, 13), (1, 320), 0); del buf1113  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf1109, buf1111, mul_78, buf1118, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_78
        buf1119 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1118, buf1119, 320, 13, grid=grid(320), stream=stream0)
        buf1120 = reinterpret_tensor(buf1118, (320, 13), (13, 1), 0); del buf1118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf1109, buf1111, buf1120, 4160, 121, grid=grid(4160), stream=stream0)
        buf1121 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf1120, buf1121, 320, 13, grid=grid(320), stream=stream0)
        buf1123 = reinterpret_tensor(buf1111, (8, 320, 14, 14), (62720, 196, 14, 1), 0); del buf1111  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_46.run(buf1122, buf1123, 2560, 196, grid=grid(2560, 196), stream=stream0)
        buf1124 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_47.run(buf1123, buf1124, 320, 1568, grid=grid(320), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1125 = aten.convolution_backward(buf1123, view_131, primals_161, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, False])
        del primals_161
        del view_131
        buf1126 = buf1125[0]
        buf1127 = buf1125[1]
        del buf1125
        buf1128 = buf1122; del buf1122  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_48.run(buf1123, buf1126, buf1128, 2560, 196, grid=grid(2560, 196), stream=stream0)
        buf1129 = reinterpret_tensor(buf1071, (1568, 1280), (1280, 1), 0); del buf1071  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1128, (1568, 320), (320, 1), 0), permute_798, out=buf1129)
        del permute_798
        buf1130 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1128, (320, 1568), (1, 320), 0), view_129, out=buf1130)
        del view_129
        buf1131 = reinterpret_tensor(buf1120, (1, 320, 13), (4160, 1, 320), 0); del buf1120  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1128, buf1131, 4160, 121, grid=grid(4160), stream=stream0)
        buf1132 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1131, buf1132, 320, 13, grid=grid(320), stream=stream0)
        buf1133 = reinterpret_tensor(buf1129, (8, 196, 1280), (250880, 1280, 1), 0); del buf1129  # reuse
        # Source Nodes: [x_143], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_25.run(buf1133, addmm_38, 2007040, grid=grid(2007040), stream=stream0)
        del addmm_38
        buf1134 = reinterpret_tensor(buf1128, (1568, 320), (320, 1), 0); del buf1128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1133, (1568, 1280), (1280, 1), 0), permute_802, out=buf1134)
        del permute_802
        buf1135 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1133, (1280, 1568), (1, 1280), 0), view_127, out=buf1135)
        del view_127
        buf1136 = buf1074; del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_26.run(buf1133, buf1136, 16640, 121, grid=grid(16640), stream=stream0)
        del buf1133
        buf1137 = reinterpret_tensor(buf1106, (1, 1280), (1280, 1), 0); del buf1106  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_27.run(buf1136, buf1137, 1280, 13, grid=grid(1280), stream=stream0)
        del buf1136
        buf1144 = reinterpret_tensor(buf1109, (8, 196, 320), (62720, 320, 1), 0); del buf1109  # reuse
        buf1145 = reinterpret_tensor(buf1091, (1568, 320), (320, 1), 0); del buf1091  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_49.run(buf1134, primals_155, mul_73, buf1123, buf1126, div_60, buf1144, buf1145, 1568, 320, grid=grid(1568), stream=stream0)
        del buf1123
        del div_60
        del primals_155
        buf1140 = reinterpret_tensor(buf1131, (320, 13), (1, 320), 0); del buf1131  # reuse
        buf1142 = buf1078; del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1134, mul_73, buf1140, buf1142, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_73
        buf1141 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1140, buf1141, 320, 13, grid=grid(320), stream=stream0)
        buf1143 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1142, buf1143, 320, 13, grid=grid(320), stream=stream0)
        buf1146 = buf1134; del buf1134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1145, permute_806, out=buf1146)
        del permute_806
        buf1147 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1145, (320, 1568), (1, 320), 0), view_125, out=buf1147)
        del view_125
        buf1148 = reinterpret_tensor(buf1142, (1, 320, 13), (4160, 1, 320), 0); del buf1142  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1145, buf1148, 4160, 121, grid=grid(4160), stream=stream0)
        buf1149 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1148, buf1149, 320, 13, grid=grid(320), stream=stream0)
        buf1150 = reinterpret_tensor(buf1145, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf1145  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(permute_82, buf1150, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del permute_82
        buf1151 = reinterpret_tensor(buf1126, (8, 5, 196, 64), (62720, 12544, 64, 1), 0); del buf1126  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_30.run(alias_48, buf1151, 40, 12544, grid=grid(40, 12544), stream=stream0)
        del alias_48
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1152 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1146, (8, 5, 196, 64), (62720, 64, 320, 1), 0), buf1150, getitem_94, getitem_95, None, buf1151, getitem_97, getitem_98, getitem_99, 0.0, [True, True, True, False])
        del buf1146
        del buf1150
        del getitem_94
        del getitem_95
        del getitem_97
        del getitem_98
        del getitem_99
        buf1153 = buf1152[0]
        buf1154 = buf1152[1]
        buf1155 = buf1152[2]
        del buf1152
        buf1156 = buf1094; del buf1094  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf1154, buf1155, buf1156, 250880, grid=grid(250880), stream=stream0)
        buf1157 = reinterpret_tensor(buf1155, (392, 320), (320, 1), 0); del buf1155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1156, (392, 640), (640, 1), 0), permute_812, out=buf1157)
        del permute_812
        buf1158 = empty((640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1156, (640, 392), (1, 640), 0), view_121, out=buf1158)
        del view_121
        buf1159 = buf1097; del buf1097  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_32.run(buf1156, buf1159, 2560, 98, grid=grid(2560), stream=stream0)
        del buf1156
        buf1160 = empty((1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf1159, buf1160, 640, 4, grid=grid(640), stream=stream0)
        del buf1159
        buf1167 = reinterpret_tensor(buf1154, (8, 49, 320), (15680, 320, 1), 0); del buf1154  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_34.run(buf1157, primals_149, mul_71, div_61, buf1167, 392, 320, grid=grid(392), stream=stream0)
        del div_61
        del primals_149
        buf1163 = buf1101; del buf1101  # reuse
        buf1165 = empty_strided((320, 4), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_35.run(buf1157, mul_71, buf1163, buf1165, 1280, 98, grid=grid(1280), stream=stream0)
        del buf1157
        del mul_71
        buf1164 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1163, buf1164, 320, 4, grid=grid(320), stream=stream0)
        del buf1163
        buf1166 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1165, buf1166, 320, 4, grid=grid(320), stream=stream0)
        buf1168 = buf1165; del buf1165  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf1167, buf1168, 1280, 98, grid=grid(1280), stream=stream0)
        buf1169 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_36.run(buf1168, buf1169, 320, 4, grid=grid(320), stream=stream0)
        del buf1168
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1170 = aten.convolution_backward(reinterpret_tensor(buf1167, (8, 320, 7, 7), (15680, 1, 2240, 320), 0), view_119, primals_147, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1167
        del primals_147
        del view_119
        buf1171 = buf1170[0]
        buf1172 = buf1170[1]
        del buf1170
        buf1173 = reinterpret_tensor(buf1151, (1568, 320), (320, 1), 0); del buf1151  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1153, (1568, 320), (320, 1), 0), permute_819, out=buf1173)
        del permute_819
        buf1174 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1153, (320, 1568), (1, 320), 0), view_116, out=buf1174)
        del view_116
        buf1175 = buf1148; del buf1148  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf1153, buf1175, 4160, 121, grid=grid(4160), stream=stream0)
        buf1176 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf1175, buf1176, 320, 13, grid=grid(320), stream=stream0)
        buf1177 = buf1115; del buf1115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_38.run(buf1171, buf1173, primals_143, buf1177, 4704, 107, grid=grid(4704), stream=stream0)
        buf1178 = buf1116; del buf1116  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_39.run(buf1177, buf1178, 1568, 3, grid=grid(1568), stream=stream0)
        del buf1177
        buf1184 = buf1144; del buf1144  # reuse
        buf1191 = reinterpret_tensor(buf1153, (8, 196, 320), (62720, 320, 1), 0); del buf1153  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_50.run(buf1184, buf1171, buf1173, primals_143, mul_69, div_62, buf1178, primals_141, mul_67, div_63, buf1191, 1568, 320, grid=grid(1568), stream=stream0)
        del buf1178
        del div_62
        del div_63
        del primals_141
        del primals_143
        buf1180 = reinterpret_tensor(buf1175, (320, 13), (1, 320), 0); del buf1175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_41.run(buf1171, buf1173, mul_69, buf1180, 4160, 121, grid=grid(4160), stream=stream0)
        del mul_69
        buf1181 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1180, buf1181, 320, 13, grid=grid(320), stream=stream0)
        buf1182 = reinterpret_tensor(buf1180, (320, 13), (13, 1), 0); del buf1180  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_42.run(buf1171, buf1173, buf1182, 4160, 121, grid=grid(4160), stream=stream0)
        del buf1171
        del buf1173
        buf1183 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_43.run(buf1182, buf1183, 320, 13, grid=grid(320), stream=stream0)
        buf1187 = reinterpret_tensor(buf1182, (320, 13), (1, 320), 0); del buf1182  # reuse
        buf1189 = buf1140; del buf1140  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_29.run(buf1184, mul_67, buf1187, buf1189, 4160, 121, grid=grid(4160), stream=stream0)
        del buf1184
        del mul_67
        buf1188 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1187, buf1188, 320, 13, grid=grid(320), stream=stream0)
        del buf1187
        buf1190 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_24.run(buf1189, buf1190, 320, 13, grid=grid(320), stream=stream0)
        buf1192 = buf1189; del buf1189  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_51.run(buf1191, buf1192, 4160, 121, grid=grid(4160), stream=stream0)
        buf1193 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_sum_24.run(buf1192, buf1193, 320, 13, grid=grid(320), stream=stream0)
        del buf1192
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1194 = aten.convolution_backward(reinterpret_tensor(buf1191, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), permute_79, primals_139, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1191
        del permute_79
        del primals_139
        buf1195 = buf1194[0]
        buf1196 = buf1194[1]
        del buf1194
        buf1197 = reinterpret_tensor(buf104, (6272, 128), (128, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_52.run(buf1195, buf1197, 6272, 128, grid=grid(6272, 128), stream=stream0)
        buf1198 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1197, permute_825, out=buf1198)
        del permute_825
        buf1199 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1197, (128, 6272), (1, 128), 0), view_112, out=buf1199)
        del view_112
        buf1200 = empty_strided((1, 128, 49), (6272, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1197, buf1200, 6272, 128, grid=grid(6272), stream=stream0)
        buf1201 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1200, buf1201, 128, 49, grid=grid(128), stream=stream0)
        buf1202 = reinterpret_tensor(buf1198, (8, 784, 1024), (802816, 1024, 1), 0); del buf1198  # reuse
        # Source Nodes: [x_122], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf1202, addmm_33, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_33
        buf1203 = buf1197; del buf1197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1202, (6272, 1024), (1024, 1), 0), permute_829, out=buf1203)
        del permute_829
        buf1204 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1202, (1024, 6272), (1, 1024), 0), view_110, out=buf1204)
        del view_110
        buf1205 = empty_strided((1, 1024, 49), (50176, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_56.run(buf1202, buf1205, 50176, 128, grid=grid(50176), stream=stream0)
        buf1206 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_57.run(buf1205, buf1206, 1024, 49, grid=grid(1024), stream=stream0)
        del buf1205
        buf1213 = empty((8, 784, 128), device='cuda', dtype=torch.float32)
        buf1214 = empty((6272, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_58.run(buf1203, primals_133, mul_62, buf1195, div_64, buf1213, buf1214, 6272, 128, grid=grid(6272), stream=stream0)
        del div_64
        del primals_133
        buf1209 = reinterpret_tensor(buf1200, (128, 49), (1, 128), 0); del buf1200  # reuse
        buf1211 = empty_strided((128, 49), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_59.run(buf1203, mul_62, buf1209, buf1211, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_62
        buf1210 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1209, buf1210, 128, 49, grid=grid(128), stream=stream0)
        buf1212 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1211, buf1212, 128, 49, grid=grid(128), stream=stream0)
        buf1215 = buf1203; del buf1203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1214, permute_833, out=buf1215)
        del permute_833
        buf1216 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1214, (128, 6272), (1, 128), 0), view_108, out=buf1216)
        del view_108
        buf1217 = reinterpret_tensor(buf1211, (1, 128, 49), (6272, 1, 128), 0); del buf1211  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1214, buf1217, 6272, 128, grid=grid(6272), stream=stream0)
        buf1218 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1217, buf1218, 128, 49, grid=grid(128), stream=stream0)
        buf1219 = reinterpret_tensor(buf1214, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1214  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(permute_70, buf1219, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del permute_70
        buf1220 = reinterpret_tensor(buf1195, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1195  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(alias_49, buf1220, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del alias_49
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1221 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1215, (8, 2, 784, 64), (100352, 64, 128, 1), 0), buf1219, getitem_80, getitem_81, None, buf1220, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False])
        del buf1215
        del buf1219
        del getitem_80
        del getitem_81
        del getitem_83
        del getitem_84
        del getitem_85
        buf1222 = buf1221[0]
        buf1223 = buf1221[1]
        buf1224 = buf1221[2]
        del buf1221
        buf1225 = empty((8, 49, 2, 2, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf1223, buf1224, buf1225, 100352, grid=grid(100352), stream=stream0)
        buf1226 = reinterpret_tensor(buf1224, (392, 128), (128, 1), 0); del buf1224  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1225, (392, 256), (256, 1), 0), permute_839, out=buf1226)
        del permute_839
        buf1227 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1225, (256, 392), (1, 256), 0), view_104, out=buf1227)
        del view_104
        buf1228 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_62.run(buf1225, buf1228, 1024, 98, grid=grid(1024), stream=stream0)
        buf1229 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_63.run(buf1228, buf1229, 256, 4, grid=grid(256), stream=stream0)
        buf1236 = reinterpret_tensor(buf1223, (8, 49, 128), (6272, 128, 1), 0); del buf1223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_64.run(buf1226, primals_127, mul_60, div_65, buf1236, 392, 128, grid=grid(392), stream=stream0)
        del div_65
        del primals_127
        buf1232 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        buf1234 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_65.run(buf1226, mul_60, buf1232, buf1234, 512, 98, grid=grid(512), stream=stream0)
        del buf1226
        del mul_60
        buf1233 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1232, buf1233, 128, 4, grid=grid(128), stream=stream0)
        buf1235 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1234, buf1235, 128, 4, grid=grid(128), stream=stream0)
        buf1237 = buf1234; del buf1234  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_67.run(buf1236, buf1237, 512, 98, grid=grid(512), stream=stream0)
        buf1238 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1237, buf1238, 128, 4, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1239 = aten.convolution_backward(reinterpret_tensor(buf1236, (8, 128, 7, 7), (6272, 1, 896, 128), 0), view_102, primals_125, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_125
        del view_102
        buf1240 = buf1239[0]
        buf1241 = buf1239[1]
        del buf1239
        buf1242 = reinterpret_tensor(buf1220, (6272, 128), (128, 1), 0); del buf1220  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1222, (6272, 128), (128, 1), 0), permute_846, out=buf1242)
        del permute_846
        buf1243 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1222, (128, 6272), (1, 128), 0), view_99, out=buf1243)
        del view_99
        buf1244 = buf1217; del buf1217  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1222, buf1244, 6272, 128, grid=grid(6272), stream=stream0)
        buf1245 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1244, buf1245, 128, 49, grid=grid(128), stream=stream0)
        buf1252 = buf1213; del buf1213  # reuse
        buf1253 = reinterpret_tensor(buf1222, (6272, 128), (128, 1), 0); del buf1222  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_red_fused__unsafe_view_add_clone_native_layer_norm_backward_68.run(buf1252, buf1240, buf1242, primals_121, mul_58, div_66, buf1253, 6272, 128, grid=grid(6272), stream=stream0)
        del div_66
        del primals_121
        buf1248 = reinterpret_tensor(buf1244, (128, 49), (1, 128), 0); del buf1244  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_69.run(buf1240, buf1242, mul_58, buf1248, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_58
        buf1249 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1248, buf1249, 128, 49, grid=grid(128), stream=stream0)
        buf1250 = reinterpret_tensor(buf1248, (128, 49), (49, 1), 0); del buf1248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_70.run(buf1240, buf1242, buf1250, 6272, 128, grid=grid(6272), stream=stream0)
        buf1251 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_71.run(buf1250, buf1251, 128, 49, grid=grid(128), stream=stream0)
        buf1254 = reinterpret_tensor(buf1202, (6272, 1024), (1024, 1), 0); del buf1202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1253, permute_850, out=buf1254)
        del permute_850
        buf1255 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1253, (128, 6272), (1, 128), 0), view_97, out=buf1255)
        del view_97
        buf1256 = reinterpret_tensor(buf1250, (1, 128, 49), (6272, 1, 128), 0); del buf1250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1253, buf1256, 6272, 128, grid=grid(6272), stream=stream0)
        buf1257 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1256, buf1257, 128, 49, grid=grid(128), stream=stream0)
        buf1258 = reinterpret_tensor(buf1254, (8, 784, 1024), (802816, 1024, 1), 0); del buf1254  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf1258, addmm_28, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_28
        buf1259 = buf1253; del buf1253  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1258, (6272, 1024), (1024, 1), 0), permute_854, out=buf1259)
        del permute_854
        buf1260 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1258, (1024, 6272), (1, 1024), 0), view_95, out=buf1260)
        del view_95
        buf1261 = reinterpret_tensor(buf1236, (1, 1024, 49), (50176, 1, 1024), 0); del buf1236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_56.run(buf1258, buf1261, 50176, 128, grid=grid(50176), stream=stream0)
        buf1262 = reinterpret_tensor(buf1228, (1, 1024), (1024, 1), 0); del buf1228  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_57.run(buf1261, buf1262, 1024, 49, grid=grid(1024), stream=stream0)
        del buf1261
        buf1269 = buf1252; del buf1252  # reuse
        buf1270 = buf1242; del buf1242  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_72.run(buf1269, buf1259, primals_115, mul_53, div_67, buf1270, 6272, 128, grid=grid(6272), stream=stream0)
        del div_67
        del primals_115
        buf1265 = reinterpret_tensor(buf1256, (128, 49), (1, 128), 0); del buf1256  # reuse
        buf1267 = buf1209; del buf1209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_59.run(buf1259, mul_53, buf1265, buf1267, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_53
        buf1266 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1265, buf1266, 128, 49, grid=grid(128), stream=stream0)
        buf1268 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1267, buf1268, 128, 49, grid=grid(128), stream=stream0)
        buf1271 = buf1259; del buf1259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1270, permute_858, out=buf1271)
        del permute_858
        buf1272 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1270, (128, 6272), (1, 128), 0), view_93, out=buf1272)
        del view_93
        buf1273 = reinterpret_tensor(buf1267, (1, 128, 49), (6272, 1, 128), 0); del buf1267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1270, buf1273, 6272, 128, grid=grid(6272), stream=stream0)
        buf1274 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1273, buf1274, 128, 49, grid=grid(128), stream=stream0)
        buf1275 = reinterpret_tensor(buf1270, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1270  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(permute_60, buf1275, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del permute_60
        buf1276 = reinterpret_tensor(buf1240, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1240  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(alias_50, buf1276, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del alias_50
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1277 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1271, (8, 2, 784, 64), (100352, 64, 128, 1), 0), buf1275, getitem_68, getitem_69, None, buf1276, getitem_71, getitem_72, getitem_73, 0.0, [True, True, True, False])
        del buf1271
        del buf1275
        del getitem_68
        del getitem_69
        del getitem_71
        del getitem_72
        del getitem_73
        buf1278 = buf1277[0]
        buf1279 = buf1277[1]
        buf1280 = buf1277[2]
        del buf1277
        buf1281 = buf1225; del buf1225  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf1279, buf1280, buf1281, 100352, grid=grid(100352), stream=stream0)
        buf1282 = reinterpret_tensor(buf1280, (392, 128), (128, 1), 0); del buf1280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1281, (392, 256), (256, 1), 0), permute_864, out=buf1282)
        del permute_864
        buf1283 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1281, (256, 392), (1, 256), 0), view_89, out=buf1283)
        del view_89
        buf1284 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_62.run(buf1281, buf1284, 1024, 98, grid=grid(1024), stream=stream0)
        buf1285 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_63.run(buf1284, buf1285, 256, 4, grid=grid(256), stream=stream0)
        buf1292 = reinterpret_tensor(buf1279, (8, 49, 128), (6272, 128, 1), 0); del buf1279  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_64.run(buf1282, primals_109, mul_51, div_68, buf1292, 392, 128, grid=grid(392), stream=stream0)
        del div_68
        del primals_109
        buf1288 = buf1237; del buf1237  # reuse
        buf1290 = buf1232; del buf1232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_65.run(buf1282, mul_51, buf1288, buf1290, 512, 98, grid=grid(512), stream=stream0)
        del buf1282
        del mul_51
        buf1289 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1288, buf1289, 128, 4, grid=grid(128), stream=stream0)
        buf1291 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1290, buf1291, 128, 4, grid=grid(128), stream=stream0)
        buf1293 = buf1290; del buf1290  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_67.run(buf1292, buf1293, 512, 98, grid=grid(512), stream=stream0)
        buf1294 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1293, buf1294, 128, 4, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1295 = aten.convolution_backward(reinterpret_tensor(buf1292, (8, 128, 7, 7), (6272, 1, 896, 128), 0), view_87, primals_107, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_107
        del view_87
        buf1296 = buf1295[0]
        buf1297 = buf1295[1]
        del buf1295
        buf1298 = reinterpret_tensor(buf1276, (6272, 128), (128, 1), 0); del buf1276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1278, (6272, 128), (128, 1), 0), permute_871, out=buf1298)
        del permute_871
        buf1299 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1278, (128, 6272), (1, 128), 0), view_84, out=buf1299)
        del view_84
        buf1300 = buf1273; del buf1273  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1278, buf1300, 6272, 128, grid=grid(6272), stream=stream0)
        buf1301 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1300, buf1301, 128, 49, grid=grid(128), stream=stream0)
        buf1308 = buf1269; del buf1269  # reuse
        buf1309 = reinterpret_tensor(buf1278, (6272, 128), (128, 1), 0); del buf1278  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_red_fused__unsafe_view_add_clone_native_layer_norm_backward_68.run(buf1308, buf1296, buf1298, primals_103, mul_49, div_69, buf1309, 6272, 128, grid=grid(6272), stream=stream0)
        del div_69
        del primals_103
        buf1304 = reinterpret_tensor(buf1300, (128, 49), (1, 128), 0); del buf1300  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_69.run(buf1296, buf1298, mul_49, buf1304, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_49
        buf1305 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1304, buf1305, 128, 49, grid=grid(128), stream=stream0)
        buf1306 = reinterpret_tensor(buf1304, (128, 49), (49, 1), 0); del buf1304  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_70.run(buf1296, buf1298, buf1306, 6272, 128, grid=grid(6272), stream=stream0)
        buf1307 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_71.run(buf1306, buf1307, 128, 49, grid=grid(128), stream=stream0)
        buf1310 = reinterpret_tensor(buf1258, (6272, 1024), (1024, 1), 0); del buf1258  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1309, permute_875, out=buf1310)
        del permute_875
        buf1311 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1309, (128, 6272), (1, 128), 0), view_82, out=buf1311)
        del view_82
        buf1312 = reinterpret_tensor(buf1306, (1, 128, 49), (6272, 1, 128), 0); del buf1306  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1309, buf1312, 6272, 128, grid=grid(6272), stream=stream0)
        buf1313 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1312, buf1313, 128, 49, grid=grid(128), stream=stream0)
        buf1314 = reinterpret_tensor(buf1310, (8, 784, 1024), (802816, 1024, 1), 0); del buf1310  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf1314, addmm_23, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_23
        buf1315 = buf1309; del buf1309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1314, (6272, 1024), (1024, 1), 0), permute_879, out=buf1315)
        del permute_879
        buf1316 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1314, (1024, 6272), (1, 1024), 0), view_80, out=buf1316)
        del view_80
        buf1317 = reinterpret_tensor(buf1292, (1, 1024, 49), (50176, 1, 1024), 0); del buf1292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_56.run(buf1314, buf1317, 50176, 128, grid=grid(50176), stream=stream0)
        buf1318 = reinterpret_tensor(buf1284, (1, 1024), (1024, 1), 0); del buf1284  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_57.run(buf1317, buf1318, 1024, 49, grid=grid(1024), stream=stream0)
        del buf1317
        buf1325 = buf1308; del buf1308  # reuse
        buf1326 = buf1298; del buf1298  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_72.run(buf1325, buf1315, primals_97, mul_44, div_70, buf1326, 6272, 128, grid=grid(6272), stream=stream0)
        del div_70
        del primals_97
        buf1321 = reinterpret_tensor(buf1312, (128, 49), (1, 128), 0); del buf1312  # reuse
        buf1323 = buf1265; del buf1265  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_59.run(buf1315, mul_44, buf1321, buf1323, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_44
        buf1322 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1321, buf1322, 128, 49, grid=grid(128), stream=stream0)
        buf1324 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1323, buf1324, 128, 49, grid=grid(128), stream=stream0)
        buf1327 = buf1315; del buf1315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1326, permute_883, out=buf1327)
        del permute_883
        buf1328 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1326, (128, 6272), (1, 128), 0), view_78, out=buf1328)
        del view_78
        buf1329 = reinterpret_tensor(buf1323, (1, 128, 49), (6272, 1, 128), 0); del buf1323  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1326, buf1329, 6272, 128, grid=grid(6272), stream=stream0)
        buf1330 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1329, buf1330, 128, 49, grid=grid(128), stream=stream0)
        buf1331 = reinterpret_tensor(buf1326, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1326  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(permute_50, buf1331, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del permute_50
        buf1332 = reinterpret_tensor(buf1296, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1296  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(alias_51, buf1332, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del alias_51
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1333 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1327, (8, 2, 784, 64), (100352, 64, 128, 1), 0), buf1331, getitem_56, getitem_57, None, buf1332, getitem_59, getitem_60, getitem_61, 0.0, [True, True, True, False])
        del buf1327
        del buf1331
        del getitem_56
        del getitem_57
        del getitem_59
        del getitem_60
        del getitem_61
        buf1334 = buf1333[0]
        buf1335 = buf1333[1]
        buf1336 = buf1333[2]
        del buf1333
        buf1337 = buf1281; del buf1281  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf1335, buf1336, buf1337, 100352, grid=grid(100352), stream=stream0)
        buf1338 = reinterpret_tensor(buf1336, (392, 128), (128, 1), 0); del buf1336  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1337, (392, 256), (256, 1), 0), permute_889, out=buf1338)
        del permute_889
        buf1339 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1337, (256, 392), (1, 256), 0), view_74, out=buf1339)
        del view_74
        buf1340 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_62.run(buf1337, buf1340, 1024, 98, grid=grid(1024), stream=stream0)
        buf1341 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_63.run(buf1340, buf1341, 256, 4, grid=grid(256), stream=stream0)
        buf1348 = reinterpret_tensor(buf1335, (8, 49, 128), (6272, 128, 1), 0); del buf1335  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_64.run(buf1338, primals_91, mul_42, div_71, buf1348, 392, 128, grid=grid(392), stream=stream0)
        del div_71
        del primals_91
        buf1344 = buf1293; del buf1293  # reuse
        buf1346 = buf1288; del buf1288  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_65.run(buf1338, mul_42, buf1344, buf1346, 512, 98, grid=grid(512), stream=stream0)
        del buf1338
        del mul_42
        buf1345 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1344, buf1345, 128, 4, grid=grid(128), stream=stream0)
        buf1347 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1346, buf1347, 128, 4, grid=grid(128), stream=stream0)
        buf1349 = buf1346; del buf1346  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_67.run(buf1348, buf1349, 512, 98, grid=grid(512), stream=stream0)
        buf1350 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1349, buf1350, 128, 4, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1351 = aten.convolution_backward(reinterpret_tensor(buf1348, (8, 128, 7, 7), (6272, 1, 896, 128), 0), view_72, primals_89, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_89
        del view_72
        buf1352 = buf1351[0]
        buf1353 = buf1351[1]
        del buf1351
        buf1354 = reinterpret_tensor(buf1332, (6272, 128), (128, 1), 0); del buf1332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1334, (6272, 128), (128, 1), 0), permute_896, out=buf1354)
        del permute_896
        buf1355 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1334, (128, 6272), (1, 128), 0), view_69, out=buf1355)
        del view_69
        buf1356 = buf1329; del buf1329  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1334, buf1356, 6272, 128, grid=grid(6272), stream=stream0)
        buf1357 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1356, buf1357, 128, 49, grid=grid(128), stream=stream0)
        buf1364 = buf1325; del buf1325  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_73.run(buf1364, buf1352, buf1354, primals_85, mul_40, div_72, 6272, 128, grid=grid(6272), stream=stream0)
        del div_72
        del primals_85
        buf1360 = reinterpret_tensor(buf1356, (128, 49), (1, 128), 0); del buf1356  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_69.run(buf1352, buf1354, mul_40, buf1360, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_40
        buf1361 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1360, buf1361, 128, 49, grid=grid(128), stream=stream0)
        buf1362 = reinterpret_tensor(buf1360, (128, 49), (49, 1), 0); del buf1360  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_70.run(buf1352, buf1354, buf1362, 6272, 128, grid=grid(6272), stream=stream0)
        buf1363 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_71.run(buf1362, buf1363, 128, 49, grid=grid(128), stream=stream0)
        buf1365 = reinterpret_tensor(buf1354, (8, 128, 28, 28), (100352, 784, 28, 1), 0); del buf1354  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_74.run(buf1364, buf1365, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf1366 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_75.run(buf1365, buf1366, 128, 6272, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1367 = aten.convolution_backward(buf1365, view_66, primals_83, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del primals_83
        del view_66
        buf1368 = buf1367[0]
        buf1369 = buf1367[1]
        del buf1367
        buf1370 = buf1364; del buf1364  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_76.run(buf1365, buf1368, buf1370, 1024, 784, grid=grid(1024, 784), stream=stream0)
        buf1371 = reinterpret_tensor(buf1314, (6272, 1024), (1024, 1), 0); del buf1314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (6272, 128), (128, 1), 0), permute_902, out=buf1371)
        del permute_902
        buf1372 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1370, (128, 6272), (1, 128), 0), view_64, out=buf1372)
        del view_64
        buf1373 = reinterpret_tensor(buf1362, (1, 128, 49), (6272, 1, 128), 0); del buf1362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1370, buf1373, 6272, 128, grid=grid(6272), stream=stream0)
        buf1374 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1373, buf1374, 128, 49, grid=grid(128), stream=stream0)
        buf1375 = reinterpret_tensor(buf1371, (8, 784, 1024), (802816, 1024, 1), 0); del buf1371  # reuse
        # Source Nodes: [x_70], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_55.run(buf1375, addmm_18, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_18
        buf1376 = reinterpret_tensor(buf1370, (6272, 128), (128, 1), 0); del buf1370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1375, (6272, 1024), (1024, 1), 0), permute_906, out=buf1376)
        del permute_906
        buf1377 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1375, (1024, 6272), (1, 1024), 0), view_62, out=buf1377)
        del view_62
        buf1378 = reinterpret_tensor(buf1348, (1, 1024, 49), (50176, 1, 1024), 0); del buf1348  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_56.run(buf1375, buf1378, 50176, 128, grid=grid(50176), stream=stream0)
        del buf1375
        buf1379 = reinterpret_tensor(buf1340, (1, 1024), (1024, 1), 0); del buf1340  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_57.run(buf1378, buf1379, 1024, 49, grid=grid(1024), stream=stream0)
        del buf1378
        buf1386 = reinterpret_tensor(buf1352, (8, 784, 128), (100352, 128, 1), 0); del buf1352  # reuse
        buf1387 = reinterpret_tensor(buf1334, (6272, 128), (128, 1), 0); del buf1334  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_77.run(buf1376, primals_77, mul_35, buf1365, buf1368, div_73, buf1386, buf1387, 6272, 128, grid=grid(6272), stream=stream0)
        del buf1365
        del div_73
        del primals_77
        buf1382 = reinterpret_tensor(buf1373, (128, 49), (1, 128), 0); del buf1373  # reuse
        buf1384 = buf1321; del buf1321  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_59.run(buf1376, mul_35, buf1382, buf1384, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_35
        buf1383 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1382, buf1383, 128, 49, grid=grid(128), stream=stream0)
        buf1385 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1384, buf1385, 128, 49, grid=grid(128), stream=stream0)
        buf1388 = buf1376; del buf1376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1387, permute_910, out=buf1388)
        del permute_910
        buf1389 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1387, (128, 6272), (1, 128), 0), view_60, out=buf1389)
        del view_60
        buf1390 = reinterpret_tensor(buf1384, (1, 128, 49), (6272, 1, 128), 0); del buf1384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1387, buf1390, 6272, 128, grid=grid(6272), stream=stream0)
        buf1391 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1390, buf1391, 128, 49, grid=grid(128), stream=stream0)
        buf1392 = reinterpret_tensor(buf1387, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1387  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(permute_37, buf1392, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del permute_37
        buf1393 = reinterpret_tensor(buf1368, (8, 2, 784, 64), (100352, 50176, 64, 1), 0); del buf1368  # reuse
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        triton_poi_fused__scaled_dot_product_efficient_attention_backward_60.run(alias_52, buf1393, 16, 50176, grid=grid(16, 50176), stream=stream0)
        del alias_52
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1394 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1388, (8, 2, 784, 64), (100352, 64, 128, 1), 0), buf1392, getitem_44, getitem_45, None, buf1393, getitem_47, getitem_48, getitem_49, 0.0, [True, True, True, False])
        del buf1388
        del buf1392
        del getitem_44
        del getitem_45
        del getitem_47
        del getitem_48
        del getitem_49
        buf1395 = buf1394[0]
        buf1396 = buf1394[1]
        buf1397 = buf1394[2]
        del buf1394
        buf1398 = buf1337; del buf1337  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_61.run(buf1396, buf1397, buf1398, 100352, grid=grid(100352), stream=stream0)
        buf1399 = reinterpret_tensor(buf1397, (392, 128), (128, 1), 0); del buf1397  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1398, (392, 256), (256, 1), 0), permute_916, out=buf1399)
        del permute_916
        buf1400 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1398, (256, 392), (1, 256), 0), view_56, out=buf1400)
        del view_56
        buf1401 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_62.run(buf1398, buf1401, 1024, 98, grid=grid(1024), stream=stream0)
        buf1402 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_63.run(buf1401, buf1402, 256, 4, grid=grid(256), stream=stream0)
        del buf1401
        buf1409 = reinterpret_tensor(buf1396, (8, 49, 128), (6272, 128, 1), 0); del buf1396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_64.run(buf1399, primals_71, mul_33, div_74, buf1409, 392, 128, grid=grid(392), stream=stream0)
        del div_74
        del primals_71
        buf1405 = buf1349; del buf1349  # reuse
        buf1407 = buf1344; del buf1344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_65.run(buf1399, mul_33, buf1405, buf1407, 512, 98, grid=grid(512), stream=stream0)
        del buf1399
        del mul_33
        buf1406 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1405, buf1406, 128, 4, grid=grid(128), stream=stream0)
        buf1408 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1407, buf1408, 128, 4, grid=grid(128), stream=stream0)
        buf1410 = buf1407; del buf1407  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_67.run(buf1409, buf1410, 512, 98, grid=grid(512), stream=stream0)
        buf1411 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_66.run(buf1410, buf1411, 128, 4, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1412 = aten.convolution_backward(reinterpret_tensor(buf1409, (8, 128, 7, 7), (6272, 1, 896, 128), 0), view_54, primals_69, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_69
        del view_54
        buf1413 = buf1412[0]
        buf1414 = buf1412[1]
        del buf1412
        buf1415 = reinterpret_tensor(buf1393, (6272, 128), (128, 1), 0); del buf1393  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1395, (6272, 128), (128, 1), 0), permute_923, out=buf1415)
        del permute_923
        buf1416 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1395, (128, 6272), (1, 128), 0), view_51, out=buf1416)
        del view_51
        buf1417 = buf1390; del buf1390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf1395, buf1417, 6272, 128, grid=grid(6272), stream=stream0)
        buf1418 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_54.run(buf1417, buf1418, 128, 49, grid=grid(128), stream=stream0)
        buf1425 = buf1386; del buf1386  # reuse
        buf1432 = reinterpret_tensor(buf1395, (8, 784, 128), (100352, 128, 1), 0); del buf1395  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_78.run(buf1425, buf1413, buf1415, primals_65, mul_31, div_75, primals_63, mul_29, div_76, buf1432, 6272, 128, grid=grid(6272), stream=stream0)
        del div_75
        del div_76
        del primals_63
        del primals_65
        buf1421 = reinterpret_tensor(buf1417, (128, 49), (1, 128), 0); del buf1417  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_69.run(buf1413, buf1415, mul_31, buf1421, 6272, 128, grid=grid(6272), stream=stream0)
        del mul_31
        buf1422 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1421, buf1422, 128, 49, grid=grid(128), stream=stream0)
        buf1423 = reinterpret_tensor(buf1421, (128, 49), (49, 1), 0); del buf1421  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_70.run(buf1413, buf1415, buf1423, 6272, 128, grid=grid(6272), stream=stream0)
        del buf1413
        del buf1415
        buf1424 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_71.run(buf1423, buf1424, 128, 49, grid=grid(128), stream=stream0)
        buf1428 = reinterpret_tensor(buf1423, (128, 49), (1, 128), 0); del buf1423  # reuse
        buf1430 = buf1382; del buf1382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_59.run(buf1425, mul_29, buf1428, buf1430, 6272, 128, grid=grid(6272), stream=stream0)
        del buf1425
        del mul_29
        buf1429 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1428, buf1429, 128, 49, grid=grid(128), stream=stream0)
        del buf1428
        buf1431 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_54.run(buf1430, buf1431, 128, 49, grid=grid(128), stream=stream0)
        buf1433 = buf1430; del buf1430  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_79.run(buf1432, buf1433, 6272, 128, grid=grid(6272), stream=stream0)
        buf1434 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_sum_54.run(buf1433, buf1434, 128, 49, grid=grid(128), stream=stream0)
        del buf1433
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1435 = aten.convolution_backward(reinterpret_tensor(buf1432, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), permute_34, primals_61, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1432
        del permute_34
        del primals_61
        buf1436 = buf1435[0]
        buf1437 = buf1435[1]
        del buf1435
        buf1438 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_80.run(buf1436, buf1438, 25088, 64, grid=grid(25088, 64), stream=stream0)
        buf1439 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1438, permute_929, out=buf1439)
        del permute_929
        buf1440 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1438, (64, 25088), (1, 64), 0), view_47, out=buf1440)
        del view_47
        buf1441 = empty_strided((1, 64, 196), (12544, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1438, buf1441, 12544, 128, grid=grid(12544), stream=stream0)
        buf1442 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1441, buf1442, 64, 196, grid=grid(64), stream=stream0)
        buf1443 = reinterpret_tensor(buf1439, (8, 3136, 512), (1605632, 512, 1), 0); del buf1439  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_83.run(buf1443, addmm_13, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_13
        buf1444 = buf1438; del buf1438  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1443, (25088, 512), (512, 1), 0), permute_933, out=buf1444)
        del permute_933
        buf1445 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1443, (512, 25088), (1, 512), 0), view_45, out=buf1445)
        del view_45
        buf1446 = reinterpret_tensor(buf1398, (1, 512, 196), (100352, 1, 512), 0); del buf1398  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf1443, buf1446, 100352, 128, grid=grid(100352), stream=stream0)
        buf1447 = reinterpret_tensor(buf1410, (1, 512), (512, 1), 0); del buf1410  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_85.run(buf1446, buf1447, 512, 196, grid=grid(512), stream=stream0)
        buf1454 = empty((8, 3136, 64), device='cuda', dtype=torch.float32)
        buf1455 = empty((25088, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_86.run(buf1444, primals_55, mul_24, buf1436, div_77, buf1454, buf1455, 25088, 64, grid=grid(25088), stream=stream0)
        del buf1436
        del div_77
        del primals_55
        buf1450 = reinterpret_tensor(buf1441, (64, 196), (1, 64), 0); del buf1441  # reuse
        buf1452 = empty_strided((64, 196), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_87.run(buf1444, mul_24, buf1450, buf1452, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_24
        buf1451 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1450, buf1451, 64, 196, grid=grid(64), stream=stream0)
        buf1453 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1452, buf1453, 64, 196, grid=grid(64), stream=stream0)
        buf1456 = buf1444; del buf1444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1455, permute_937, out=buf1456)
        del permute_937
        buf1457 = reinterpret_tensor(buf129, (64, 64), (64, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1455, (64, 25088), (1, 64), 0), view_43, out=buf1457)
        del view_43
        buf1458 = reinterpret_tensor(buf1452, (1, 64, 196), (12544, 1, 64), 0); del buf1452  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1455, buf1458, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1455
        buf1459 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1458, buf1459, 64, 196, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1460 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1456, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), permute_25, getitem_30, getitem_31, None, alias_53, getitem_33, getitem_34, getitem_35, 0.0, [True, True, True, False])
        del alias_53
        del getitem_30
        del getitem_31
        del getitem_33
        del getitem_34
        del getitem_35
        del permute_25
        buf1461 = buf1460[0]
        buf1462 = buf1460[1]
        buf1463 = buf1460[2]
        del buf1460
        buf1464 = reinterpret_tensor(buf1409, (8, 49, 2, 1, 64), (6272, 128, 64, 64, 1), 0); del buf1409  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_88.run(buf1462, buf1463, buf1464, 50176, grid=grid(50176), stream=stream0)
        buf1465 = reinterpret_tensor(buf1463, (392, 64), (64, 1), 0); del buf1463  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1464, (392, 128), (128, 1), 0), permute_943, out=buf1465)
        del permute_943
        buf1466 = reinterpret_tensor(buf107, (128, 64), (64, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1464, (128, 392), (1, 128), 0), view_39, out=buf1466)
        del view_39
        buf1467 = reinterpret_tensor(buf1405, (1, 128, 4), (512, 1, 128), 0); del buf1405  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_89.run(buf1464, buf1467, 512, 98, grid=grid(512), stream=stream0)
        buf1468 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_66.run(buf1467, buf1468, 128, 4, grid=grid(128), stream=stream0)
        buf1475 = reinterpret_tensor(buf1462, (8, 49, 64), (3136, 64, 1), 0); del buf1462  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_90.run(buf1465, primals_49, mul_22, div_78, buf1475, 392, 64, grid=grid(392), stream=stream0)
        del div_78
        del primals_49
        buf1471 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        buf1473 = empty_strided((64, 4), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_91.run(buf1465, mul_22, buf1471, buf1473, 256, 98, grid=grid(256), stream=stream0)
        del buf1465
        del mul_22
        buf1472 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1471, buf1472, 64, 4, grid=grid(64), stream=stream0)
        buf1474 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1473, buf1474, 64, 4, grid=grid(64), stream=stream0)
        buf1476 = buf1473; del buf1473  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_93.run(buf1475, buf1476, 256, 98, grid=grid(256), stream=stream0)
        buf1477 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1476, buf1477, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1478 = aten.convolution_backward(reinterpret_tensor(buf1475, (8, 64, 7, 7), (3136, 1, 448, 64), 0), view_37, primals_47, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1475
        del primals_47
        del view_37
        buf1479 = buf1478[0]
        buf1480 = buf1478[1]
        del buf1478
        buf1481 = buf1456; del buf1456  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1461, (25088, 64), (64, 1), 0), permute_950, out=buf1481)
        del permute_950
        buf1482 = empty((64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1461, (64, 25088), (1, 64), 0), view_34, out=buf1482)
        del view_34
        buf1483 = buf1458; del buf1458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1461, buf1483, 12544, 128, grid=grid(12544), stream=stream0)
        buf1484 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1483, buf1484, 64, 196, grid=grid(64), stream=stream0)
        buf1491 = buf1454; del buf1454  # reuse
        buf1492 = reinterpret_tensor(buf1461, (25088, 64), (64, 1), 0); del buf1461  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_94.run(buf1491, buf1479, buf1481, primals_43, mul_20, div_79, buf1492, 25088, 64, grid=grid(25088), stream=stream0)
        del div_79
        del primals_43
        buf1487 = reinterpret_tensor(buf1483, (64, 196), (1, 64), 0); del buf1483  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_95.run(buf1479, buf1481, mul_20, buf1487, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_20
        buf1488 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1487, buf1488, 64, 196, grid=grid(64), stream=stream0)
        buf1489 = reinterpret_tensor(buf1487, (64, 196), (196, 1), 0); del buf1487  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_96.run(buf1479, buf1481, buf1489, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1479
        buf1490 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_97.run(buf1489, buf1490, 64, 196, grid=grid(64), stream=stream0)
        buf1493 = reinterpret_tensor(buf1443, (25088, 512), (512, 1), 0); del buf1443  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1492, permute_954, out=buf1493)
        del permute_954
        buf1494 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1492, (64, 25088), (1, 64), 0), view_32, out=buf1494)
        del view_32
        buf1495 = reinterpret_tensor(buf1489, (1, 64, 196), (12544, 1, 64), 0); del buf1489  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1492, buf1495, 12544, 128, grid=grid(12544), stream=stream0)
        buf1496 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1495, buf1496, 64, 196, grid=grid(64), stream=stream0)
        buf1497 = reinterpret_tensor(buf1493, (8, 3136, 512), (1605632, 512, 1), 0); del buf1493  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_83.run(buf1497, addmm_8, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_8
        buf1498 = buf1492; del buf1492  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1497, (25088, 512), (512, 1), 0), permute_958, out=buf1498)
        del permute_958
        buf1499 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1497, (512, 25088), (1, 512), 0), view_30, out=buf1499)
        del view_30
        buf1500 = buf1446; del buf1446  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf1497, buf1500, 100352, 128, grid=grid(100352), stream=stream0)
        buf1501 = reinterpret_tensor(buf1467, (1, 512), (512, 1), 0); del buf1467  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_85.run(buf1500, buf1501, 512, 196, grid=grid(512), stream=stream0)
        buf1508 = buf1491; del buf1491  # reuse
        buf1509 = buf1481; del buf1481  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_98.run(buf1508, buf1498, primals_37, mul_15, div_80, buf1509, 25088, 64, grid=grid(25088), stream=stream0)
        del div_80
        del primals_37
        buf1504 = reinterpret_tensor(buf1495, (64, 196), (1, 64), 0); del buf1495  # reuse
        buf1506 = buf1450; del buf1450  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_87.run(buf1498, mul_15, buf1504, buf1506, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_15
        buf1505 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1504, buf1505, 64, 196, grid=grid(64), stream=stream0)
        buf1507 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1506, buf1507, 64, 196, grid=grid(64), stream=stream0)
        buf1510 = buf1498; del buf1498  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1509, permute_962, out=buf1510)
        del permute_962
        buf1511 = empty((64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1509, (64, 25088), (1, 64), 0), view_28, out=buf1511)
        del view_28
        buf1512 = reinterpret_tensor(buf1506, (1, 64, 196), (12544, 1, 64), 0); del buf1506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1509, buf1512, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1509
        buf1513 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1512, buf1513, 64, 196, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1514 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1510, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), permute_15, getitem_18, getitem_19, None, alias_54, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False])
        del alias_54
        del getitem_18
        del getitem_19
        del getitem_21
        del getitem_22
        del getitem_23
        del permute_15
        buf1515 = buf1514[0]
        buf1516 = buf1514[1]
        buf1517 = buf1514[2]
        del buf1514
        buf1518 = buf1464; del buf1464  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_88.run(buf1516, buf1517, buf1518, 50176, grid=grid(50176), stream=stream0)
        buf1519 = reinterpret_tensor(buf1517, (392, 64), (64, 1), 0); del buf1517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1518, (392, 128), (128, 1), 0), permute_968, out=buf1519)
        del permute_968
        buf1520 = empty((128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1518, (128, 392), (1, 128), 0), view_24, out=buf1520)
        del view_24
        buf1521 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_89.run(buf1518, buf1521, 512, 98, grid=grid(512), stream=stream0)
        buf1522 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_66.run(buf1521, buf1522, 128, 4, grid=grid(128), stream=stream0)
        buf1529 = reinterpret_tensor(buf1516, (8, 49, 64), (3136, 64, 1), 0); del buf1516  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_90.run(buf1519, primals_31, mul_13, div_81, buf1529, 392, 64, grid=grid(392), stream=stream0)
        del div_81
        del primals_31
        buf1525 = buf1476; del buf1476  # reuse
        buf1527 = buf1471; del buf1471  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_91.run(buf1519, mul_13, buf1525, buf1527, 256, 98, grid=grid(256), stream=stream0)
        del buf1519
        del mul_13
        buf1526 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1525, buf1526, 64, 4, grid=grid(64), stream=stream0)
        buf1528 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1527, buf1528, 64, 4, grid=grid(64), stream=stream0)
        buf1530 = buf1527; del buf1527  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_93.run(buf1529, buf1530, 256, 98, grid=grid(256), stream=stream0)
        buf1531 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1530, buf1531, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1532 = aten.convolution_backward(reinterpret_tensor(buf1529, (8, 64, 7, 7), (3136, 1, 448, 64), 0), view_22, primals_29, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1529
        del primals_29
        del view_22
        buf1533 = buf1532[0]
        buf1534 = buf1532[1]
        del buf1532
        buf1535 = buf1510; del buf1510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1515, (25088, 64), (64, 1), 0), permute_975, out=buf1535)
        del permute_975
        buf1536 = empty((64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1515, (64, 25088), (1, 64), 0), view_19, out=buf1536)
        del view_19
        buf1537 = buf1512; del buf1512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1515, buf1537, 12544, 128, grid=grid(12544), stream=stream0)
        buf1538 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1537, buf1538, 64, 196, grid=grid(64), stream=stream0)
        buf1545 = buf1508; del buf1508  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_99.run(buf1545, buf1533, buf1535, primals_25, mul_11, div_82, 25088, 64, grid=grid(25088), stream=stream0)
        del div_82
        del primals_25
        buf1541 = reinterpret_tensor(buf1537, (64, 196), (1, 64), 0); del buf1537  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_95.run(buf1533, buf1535, mul_11, buf1541, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_11
        buf1542 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1541, buf1542, 64, 196, grid=grid(64), stream=stream0)
        buf1543 = reinterpret_tensor(buf1541, (64, 196), (196, 1), 0); del buf1541  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_96.run(buf1533, buf1535, buf1543, 12544, 128, grid=grid(12544), stream=stream0)
        buf1544 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_97.run(buf1543, buf1544, 64, 196, grid=grid(64), stream=stream0)
        buf1546 = reinterpret_tensor(buf1535, (8, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1535  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_100.run(buf1545, buf1546, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf1547 = buf1530; del buf1530  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_101.run(buf1546, buf1547, 256, 6272, grid=grid(256), stream=stream0)
        buf1548 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_102.run(buf1547, buf1548, 64, 4, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1549 = aten.convolution_backward(buf1546, view_16, primals_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del primals_23
        del view_16
        buf1550 = buf1549[0]
        buf1551 = buf1549[1]
        del buf1549
        buf1552 = buf1545; del buf1545  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_103.run(buf1546, buf1550, buf1552, 512, 3136, grid=grid(512, 3136), stream=stream0)
        buf1553 = reinterpret_tensor(buf1497, (25088, 512), (512, 1), 0); del buf1497  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1552, (25088, 64), (64, 1), 0), permute_981, out=buf1553)
        del permute_981
        buf1554 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1552, (64, 25088), (1, 64), 0), view_14, out=buf1554)
        del view_14
        buf1555 = reinterpret_tensor(buf1543, (1, 64, 196), (12544, 1, 64), 0); del buf1543  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1552, buf1555, 12544, 128, grid=grid(12544), stream=stream0)
        buf1556 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1555, buf1556, 64, 196, grid=grid(64), stream=stream0)
        buf1557 = reinterpret_tensor(buf1553, (8, 3136, 512), (1605632, 512, 1), 0); del buf1553  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_83.run(buf1557, addmm_3, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_3
        buf1558 = reinterpret_tensor(buf1552, (25088, 64), (64, 1), 0); del buf1552  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1557, (25088, 512), (512, 1), 0), permute_985, out=buf1558)
        del permute_985
        buf1559 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1557, (512, 25088), (1, 512), 0), view_12, out=buf1559)
        del view_12
        buf1560 = buf1500; del buf1500  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_84.run(buf1557, buf1560, 100352, 128, grid=grid(100352), stream=stream0)
        del buf1557
        buf1561 = reinterpret_tensor(buf1521, (1, 512), (512, 1), 0); del buf1521  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_85.run(buf1560, buf1561, 512, 196, grid=grid(512), stream=stream0)
        del buf1560
        buf1568 = reinterpret_tensor(buf1533, (8, 3136, 64), (200704, 64, 1), 0); del buf1533  # reuse
        buf1569 = reinterpret_tensor(buf1515, (25088, 64), (64, 1), 0); del buf1515  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_104.run(buf1558, primals_17, mul_6, buf1546, buf1550, div_83, buf1568, buf1569, 25088, 64, grid=grid(25088), stream=stream0)
        del buf1546
        del buf1550
        del div_83
        del primals_17
        buf1564 = reinterpret_tensor(buf1555, (64, 196), (1, 64), 0); del buf1555  # reuse
        buf1566 = buf1504; del buf1504  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_87.run(buf1558, mul_6, buf1564, buf1566, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_6
        buf1565 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1564, buf1565, 64, 196, grid=grid(64), stream=stream0)
        buf1567 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1566, buf1567, 64, 196, grid=grid(64), stream=stream0)
        buf1570 = buf1558; del buf1558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1569, permute_989, out=buf1570)
        del permute_989
        buf1571 = empty((64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1569, (64, 25088), (1, 64), 0), view_10, out=buf1571)
        del view_10
        buf1572 = reinterpret_tensor(buf1566, (1, 64, 196), (12544, 1, 64), 0); del buf1566  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1569, buf1572, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1569
        buf1573 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1572, buf1573, 64, 196, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf1574 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf1570, (8, 1, 3136, 64), (200704, 64, 64, 1), 0), permute_2, getitem_6, getitem_7, None, alias_55, getitem_9, getitem_10, getitem_11, 0.0, [True, True, True, False])
        del alias_55
        del getitem_10
        del getitem_11
        del getitem_6
        del getitem_7
        del getitem_9
        del permute_2
        buf1575 = buf1574[0]
        buf1576 = buf1574[1]
        buf1577 = buf1574[2]
        del buf1574
        buf1578 = buf1518; del buf1518  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_88.run(buf1576, buf1577, buf1578, 50176, grid=grid(50176), stream=stream0)
        buf1579 = reinterpret_tensor(buf1577, (392, 64), (64, 1), 0); del buf1577  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (392, 128), (128, 1), 0), permute_995, out=buf1579)
        del permute_995
        buf1580 = empty((128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1578, (128, 392), (1, 128), 0), view_6, out=buf1580)
        del view_6
        buf1581 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_89.run(buf1578, buf1581, 512, 98, grid=grid(512), stream=stream0)
        del buf1578
        buf1582 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_66.run(buf1581, buf1582, 128, 4, grid=grid(128), stream=stream0)
        del buf1581
        buf1589 = reinterpret_tensor(buf1576, (8, 49, 64), (3136, 64, 1), 0); del buf1576  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_90.run(buf1579, primals_11, mul_4, div_84, buf1589, 392, 64, grid=grid(392), stream=stream0)
        del div_84
        del primals_11
        buf1585 = buf1547; del buf1547  # reuse
        buf1587 = buf1525; del buf1525  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_91.run(buf1579, mul_4, buf1585, buf1587, 256, 98, grid=grid(256), stream=stream0)
        del buf1579
        del mul_4
        buf1586 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1585, buf1586, 64, 4, grid=grid(64), stream=stream0)
        del buf1585
        buf1588 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1587, buf1588, 64, 4, grid=grid(64), stream=stream0)
        buf1590 = buf1587; del buf1587  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_93.run(buf1589, buf1590, 256, 98, grid=grid(256), stream=stream0)
        buf1591 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_92.run(buf1590, buf1591, 64, 4, grid=grid(64), stream=stream0)
        del buf1590
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1592 = aten.convolution_backward(reinterpret_tensor(buf1589, (8, 64, 7, 7), (3136, 1, 448, 64), 0), view_4, primals_9, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf1589
        del primals_9
        del view_4
        buf1593 = buf1592[0]
        buf1594 = buf1592[1]
        del buf1592
        buf1595 = buf1570; del buf1570  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1575, (25088, 64), (64, 1), 0), permute_1002, out=buf1595)
        del permute_1002
        buf1596 = empty((64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1575, (64, 25088), (1, 64), 0), view_1, out=buf1596)
        del view_1
        buf1597 = buf1572; del buf1572  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_81.run(buf1575, buf1597, 12544, 128, grid=grid(12544), stream=stream0)
        buf1598 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_82.run(buf1597, buf1598, 64, 196, grid=grid(64), stream=stream0)
        buf1605 = buf1568; del buf1568  # reuse
        buf1612 = reinterpret_tensor(buf1575, (8, 3136, 64), (200704, 64, 1), 0); del buf1575  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_105.run(buf1605, buf1593, buf1595, primals_5, mul_2, div_85, primals_3, mul, div_86, buf1612, 25088, 64, grid=grid(25088), stream=stream0)
        del div_85
        del div_86
        del primals_3
        del primals_5
        buf1601 = reinterpret_tensor(buf1597, (64, 196), (1, 64), 0); del buf1597  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_95.run(buf1593, buf1595, mul_2, buf1601, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_2
        buf1602 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1601, buf1602, 64, 196, grid=grid(64), stream=stream0)
        buf1603 = reinterpret_tensor(buf1601, (64, 196), (196, 1), 0); del buf1601  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_96.run(buf1593, buf1595, buf1603, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1593
        del buf1595
        buf1604 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_97.run(buf1603, buf1604, 64, 196, grid=grid(64), stream=stream0)
        buf1608 = reinterpret_tensor(buf1603, (64, 196), (1, 64), 0); del buf1603  # reuse
        buf1610 = buf1564; del buf1564  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_87.run(buf1605, mul, buf1608, buf1610, 12544, 128, grid=grid(12544), stream=stream0)
        del buf1605
        del mul
        buf1609 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1608, buf1609, 64, 196, grid=grid(64), stream=stream0)
        del buf1608
        buf1611 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_82.run(buf1610, buf1611, 64, 196, grid=grid(64), stream=stream0)
        buf1613 = buf1610; del buf1610  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_106.run(buf1612, buf1613, 12544, 128, grid=grid(12544), stream=stream0)
        buf1614 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_82.run(buf1613, buf1614, 64, 196, grid=grid(64), stream=stream0)
        del buf1613
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1615 = aten.convolution_backward(reinterpret_tensor(buf1612, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), primals_521, primals_1, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf1612
        del primals_1
        del primals_521
        buf1616 = buf1615[1]
        return (buf1616, buf1614, buf1609, buf1611, buf1602, buf1604, reinterpret_tensor(buf1596, (64, 64), (64, 1), 0), reinterpret_tensor(buf1598, (64, ), (1, ), 0), buf1594, buf1591, buf1586, buf1588, reinterpret_tensor(buf1580, (128, 64), (64, 1), 0), reinterpret_tensor(buf1582, (128, ), (1, ), 0), reinterpret_tensor(buf1571, (64, 64), (64, 1), 0), reinterpret_tensor(buf1573, (64, ), (1, ), 0), buf1565, buf1567, reinterpret_tensor(buf1559, (512, 64), (64, 1), 0), reinterpret_tensor(buf1561, (512, ), (1, ), 0), reinterpret_tensor(buf1554, (64, 512), (512, 1), 0), reinterpret_tensor(buf1556, (64, ), (1, ), 0), buf1551, buf1548, buf1542, buf1544, reinterpret_tensor(buf1536, (64, 64), (64, 1), 0), reinterpret_tensor(buf1538, (64, ), (1, ), 0), buf1534, buf1531, buf1526, buf1528, reinterpret_tensor(buf1520, (128, 64), (64, 1), 0), reinterpret_tensor(buf1522, (128, ), (1, ), 0), reinterpret_tensor(buf1511, (64, 64), (64, 1), 0), reinterpret_tensor(buf1513, (64, ), (1, ), 0), buf1505, buf1507, reinterpret_tensor(buf1499, (512, 64), (64, 1), 0), reinterpret_tensor(buf1501, (512, ), (1, ), 0), reinterpret_tensor(buf1494, (64, 512), (512, 1), 0), reinterpret_tensor(buf1496, (64, ), (1, ), 0), buf1488, buf1490, reinterpret_tensor(buf1482, (64, 64), (64, 1), 0), reinterpret_tensor(buf1484, (64, ), (1, ), 0), buf1480, buf1477, buf1472, buf1474, reinterpret_tensor(buf1466, (128, 64), (64, 1), 0), reinterpret_tensor(buf1468, (128, ), (1, ), 0), reinterpret_tensor(buf1457, (64, 64), (64, 1), 0), reinterpret_tensor(buf1459, (64, ), (1, ), 0), buf1451, buf1453, reinterpret_tensor(buf1445, (512, 64), (64, 1), 0), reinterpret_tensor(buf1447, (512, ), (1, ), 0), reinterpret_tensor(buf1440, (64, 512), (512, 1), 0), reinterpret_tensor(buf1442, (64, ), (1, ), 0), buf1437, buf1434, buf1429, buf1431, buf1422, buf1424, reinterpret_tensor(buf1416, (128, 128), (128, 1), 0), reinterpret_tensor(buf1418, (128, ), (1, ), 0), buf1414, buf1411, buf1406, buf1408, reinterpret_tensor(buf1400, (256, 128), (128, 1), 0), reinterpret_tensor(buf1402, (256, ), (1, ), 0), reinterpret_tensor(buf1389, (128, 128), (128, 1), 0), reinterpret_tensor(buf1391, (128, ), (1, ), 0), buf1383, buf1385, reinterpret_tensor(buf1377, (1024, 128), (128, 1), 0), reinterpret_tensor(buf1379, (1024, ), (1, ), 0), reinterpret_tensor(buf1372, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf1374, (128, ), (1, ), 0), buf1369, buf1366, buf1361, buf1363, reinterpret_tensor(buf1355, (128, 128), (128, 1), 0), reinterpret_tensor(buf1357, (128, ), (1, ), 0), buf1353, buf1350, buf1345, buf1347, reinterpret_tensor(buf1339, (256, 128), (128, 1), 0), reinterpret_tensor(buf1341, (256, ), (1, ), 0), reinterpret_tensor(buf1328, (128, 128), (128, 1), 0), reinterpret_tensor(buf1330, (128, ), (1, ), 0), buf1322, buf1324, reinterpret_tensor(buf1316, (1024, 128), (128, 1), 0), reinterpret_tensor(buf1318, (1024, ), (1, ), 0), reinterpret_tensor(buf1311, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf1313, (128, ), (1, ), 0), buf1305, buf1307, reinterpret_tensor(buf1299, (128, 128), (128, 1), 0), reinterpret_tensor(buf1301, (128, ), (1, ), 0), buf1297, buf1294, buf1289, buf1291, reinterpret_tensor(buf1283, (256, 128), (128, 1), 0), reinterpret_tensor(buf1285, (256, ), (1, ), 0), reinterpret_tensor(buf1272, (128, 128), (128, 1), 0), reinterpret_tensor(buf1274, (128, ), (1, ), 0), buf1266, buf1268, reinterpret_tensor(buf1260, (1024, 128), (128, 1), 0), reinterpret_tensor(buf1262, (1024, ), (1, ), 0), reinterpret_tensor(buf1255, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf1257, (128, ), (1, ), 0), buf1249, buf1251, reinterpret_tensor(buf1243, (128, 128), (128, 1), 0), reinterpret_tensor(buf1245, (128, ), (1, ), 0), buf1241, buf1238, buf1233, buf1235, reinterpret_tensor(buf1227, (256, 128), (128, 1), 0), reinterpret_tensor(buf1229, (256, ), (1, ), 0), reinterpret_tensor(buf1216, (128, 128), (128, 1), 0), reinterpret_tensor(buf1218, (128, ), (1, ), 0), buf1210, buf1212, reinterpret_tensor(buf1204, (1024, 128), (128, 1), 0), reinterpret_tensor(buf1206, (1024, ), (1, ), 0), reinterpret_tensor(buf1199, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf1201, (128, ), (1, ), 0), buf1196, buf1193, buf1188, buf1190, buf1181, buf1183, reinterpret_tensor(buf1174, (320, 320), (320, 1), 0), reinterpret_tensor(buf1176, (320, ), (1, ), 0), buf1172, buf1169, buf1164, buf1166, reinterpret_tensor(buf1158, (640, 320), (320, 1), 0), reinterpret_tensor(buf1160, (640, ), (1, ), 0), reinterpret_tensor(buf1147, (320, 320), (320, 1), 0), reinterpret_tensor(buf1149, (320, ), (1, ), 0), buf1141, buf1143, reinterpret_tensor(buf1135, (1280, 320), (320, 1), 0), reinterpret_tensor(buf1137, (1280, ), (1, ), 0), reinterpret_tensor(buf1130, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf1132, (320, ), (1, ), 0), buf1127, buf1124, buf1119, buf1121, reinterpret_tensor(buf1112, (320, 320), (320, 1), 0), reinterpret_tensor(buf1114, (320, ), (1, ), 0), buf1110, buf1107, buf1102, buf1104, reinterpret_tensor(buf1096, (640, 320), (320, 1), 0), reinterpret_tensor(buf1098, (640, ), (1, ), 0), reinterpret_tensor(buf1085, (320, 320), (320, 1), 0), reinterpret_tensor(buf1087, (320, ), (1, ), 0), buf1079, buf1081, reinterpret_tensor(buf1073, (1280, 320), (320, 1), 0), reinterpret_tensor(buf1075, (1280, ), (1, ), 0), reinterpret_tensor(buf1068, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf1070, (320, ), (1, ), 0), buf1062, buf1064, reinterpret_tensor(buf1055, (320, 320), (320, 1), 0), reinterpret_tensor(buf1057, (320, ), (1, ), 0), buf1053, buf1050, buf1045, buf1047, reinterpret_tensor(buf1039, (640, 320), (320, 1), 0), reinterpret_tensor(buf1041, (640, ), (1, ), 0), reinterpret_tensor(buf1028, (320, 320), (320, 1), 0), reinterpret_tensor(buf1030, (320, ), (1, ), 0), buf1022, buf1024, reinterpret_tensor(buf1016, (1280, 320), (320, 1), 0), reinterpret_tensor(buf1018, (1280, ), (1, ), 0), reinterpret_tensor(buf1011, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf1013, (320, ), (1, ), 0), buf1005, buf1007, reinterpret_tensor(buf998, (320, 320), (320, 1), 0), reinterpret_tensor(buf1000, (320, ), (1, ), 0), buf996, buf993, buf988, buf990, reinterpret_tensor(buf982, (640, 320), (320, 1), 0), reinterpret_tensor(buf984, (640, ), (1, ), 0), reinterpret_tensor(buf971, (320, 320), (320, 1), 0), reinterpret_tensor(buf973, (320, ), (1, ), 0), buf965, buf967, reinterpret_tensor(buf959, (1280, 320), (320, 1), 0), reinterpret_tensor(buf961, (1280, ), (1, ), 0), reinterpret_tensor(buf954, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf956, (320, ), (1, ), 0), buf948, buf950, reinterpret_tensor(buf941, (320, 320), (320, 1), 0), reinterpret_tensor(buf943, (320, ), (1, ), 0), buf939, buf936, buf931, buf933, reinterpret_tensor(buf925, (640, 320), (320, 1), 0), reinterpret_tensor(buf927, (640, ), (1, ), 0), reinterpret_tensor(buf914, (320, 320), (320, 1), 0), reinterpret_tensor(buf916, (320, ), (1, ), 0), buf908, buf910, reinterpret_tensor(buf902, (1280, 320), (320, 1), 0), reinterpret_tensor(buf904, (1280, ), (1, ), 0), reinterpret_tensor(buf897, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf899, (320, ), (1, ), 0), buf891, buf893, reinterpret_tensor(buf884, (320, 320), (320, 1), 0), reinterpret_tensor(buf886, (320, ), (1, ), 0), buf882, buf879, buf874, buf876, reinterpret_tensor(buf868, (640, 320), (320, 1), 0), reinterpret_tensor(buf870, (640, ), (1, ), 0), reinterpret_tensor(buf857, (320, 320), (320, 1), 0), reinterpret_tensor(buf859, (320, ), (1, ), 0), buf851, buf853, reinterpret_tensor(buf845, (1280, 320), (320, 1), 0), reinterpret_tensor(buf847, (1280, ), (1, ), 0), reinterpret_tensor(buf840, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf842, (320, ), (1, ), 0), buf834, buf836, reinterpret_tensor(buf827, (320, 320), (320, 1), 0), reinterpret_tensor(buf829, (320, ), (1, ), 0), buf825, buf822, buf817, buf819, reinterpret_tensor(buf811, (640, 320), (320, 1), 0), reinterpret_tensor(buf813, (640, ), (1, ), 0), reinterpret_tensor(buf800, (320, 320), (320, 1), 0), reinterpret_tensor(buf802, (320, ), (1, ), 0), buf794, buf796, reinterpret_tensor(buf788, (1280, 320), (320, 1), 0), reinterpret_tensor(buf790, (1280, ), (1, ), 0), reinterpret_tensor(buf783, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf785, (320, ), (1, ), 0), buf777, buf779, reinterpret_tensor(buf770, (320, 320), (320, 1), 0), reinterpret_tensor(buf772, (320, ), (1, ), 0), buf768, buf765, buf760, buf762, reinterpret_tensor(buf754, (640, 320), (320, 1), 0), reinterpret_tensor(buf756, (640, ), (1, ), 0), reinterpret_tensor(buf743, (320, 320), (320, 1), 0), reinterpret_tensor(buf745, (320, ), (1, ), 0), buf737, buf739, reinterpret_tensor(buf731, (1280, 320), (320, 1), 0), reinterpret_tensor(buf733, (1280, ), (1, ), 0), reinterpret_tensor(buf726, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf728, (320, ), (1, ), 0), buf720, buf722, reinterpret_tensor(buf713, (320, 320), (320, 1), 0), reinterpret_tensor(buf715, (320, ), (1, ), 0), buf711, buf708, buf703, buf705, reinterpret_tensor(buf697, (640, 320), (320, 1), 0), reinterpret_tensor(buf699, (640, ), (1, ), 0), reinterpret_tensor(buf686, (320, 320), (320, 1), 0), reinterpret_tensor(buf688, (320, ), (1, ), 0), buf680, buf682, reinterpret_tensor(buf674, (1280, 320), (320, 1), 0), reinterpret_tensor(buf676, (1280, ), (1, ), 0), reinterpret_tensor(buf669, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf671, (320, ), (1, ), 0), buf663, buf665, reinterpret_tensor(buf656, (320, 320), (320, 1), 0), reinterpret_tensor(buf658, (320, ), (1, ), 0), buf654, buf651, buf646, buf648, reinterpret_tensor(buf640, (640, 320), (320, 1), 0), reinterpret_tensor(buf642, (640, ), (1, ), 0), reinterpret_tensor(buf629, (320, 320), (320, 1), 0), reinterpret_tensor(buf631, (320, ), (1, ), 0), buf623, buf625, reinterpret_tensor(buf617, (1280, 320), (320, 1), 0), reinterpret_tensor(buf619, (1280, ), (1, ), 0), reinterpret_tensor(buf612, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf614, (320, ), (1, ), 0), buf606, buf608, reinterpret_tensor(buf599, (320, 320), (320, 1), 0), reinterpret_tensor(buf601, (320, ), (1, ), 0), buf597, buf594, buf589, buf591, reinterpret_tensor(buf583, (640, 320), (320, 1), 0), reinterpret_tensor(buf585, (640, ), (1, ), 0), reinterpret_tensor(buf572, (320, 320), (320, 1), 0), reinterpret_tensor(buf574, (320, ), (1, ), 0), buf566, buf568, reinterpret_tensor(buf560, (1280, 320), (320, 1), 0), reinterpret_tensor(buf562, (1280, ), (1, ), 0), reinterpret_tensor(buf555, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf557, (320, ), (1, ), 0), buf549, buf551, reinterpret_tensor(buf542, (320, 320), (320, 1), 0), reinterpret_tensor(buf544, (320, ), (1, ), 0), buf540, buf537, buf532, buf534, reinterpret_tensor(buf526, (640, 320), (320, 1), 0), reinterpret_tensor(buf528, (640, ), (1, ), 0), reinterpret_tensor(buf515, (320, 320), (320, 1), 0), reinterpret_tensor(buf517, (320, ), (1, ), 0), buf509, buf511, reinterpret_tensor(buf503, (1280, 320), (320, 1), 0), reinterpret_tensor(buf505, (1280, ), (1, ), 0), reinterpret_tensor(buf498, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf500, (320, ), (1, ), 0), buf492, buf494, reinterpret_tensor(buf485, (320, 320), (320, 1), 0), reinterpret_tensor(buf487, (320, ), (1, ), 0), buf483, buf480, buf475, buf477, reinterpret_tensor(buf469, (640, 320), (320, 1), 0), reinterpret_tensor(buf471, (640, ), (1, ), 0), reinterpret_tensor(buf458, (320, 320), (320, 1), 0), reinterpret_tensor(buf460, (320, ), (1, ), 0), buf452, buf454, reinterpret_tensor(buf446, (1280, 320), (320, 1), 0), reinterpret_tensor(buf448, (1280, ), (1, ), 0), reinterpret_tensor(buf441, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf443, (320, ), (1, ), 0), buf435, buf437, reinterpret_tensor(buf428, (320, 320), (320, 1), 0), reinterpret_tensor(buf430, (320, ), (1, ), 0), buf426, buf423, buf418, buf420, reinterpret_tensor(buf412, (640, 320), (320, 1), 0), reinterpret_tensor(buf414, (640, ), (1, ), 0), reinterpret_tensor(buf401, (320, 320), (320, 1), 0), reinterpret_tensor(buf403, (320, ), (1, ), 0), buf395, buf397, reinterpret_tensor(buf389, (1280, 320), (320, 1), 0), reinterpret_tensor(buf391, (1280, ), (1, ), 0), reinterpret_tensor(buf384, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf386, (320, ), (1, ), 0), buf378, buf380, reinterpret_tensor(buf371, (320, 320), (320, 1), 0), reinterpret_tensor(buf373, (320, ), (1, ), 0), buf369, buf366, buf361, buf363, reinterpret_tensor(buf355, (640, 320), (320, 1), 0), reinterpret_tensor(buf357, (640, ), (1, ), 0), reinterpret_tensor(buf344, (320, 320), (320, 1), 0), reinterpret_tensor(buf346, (320, ), (1, ), 0), buf338, buf340, reinterpret_tensor(buf332, (1280, 320), (320, 1), 0), reinterpret_tensor(buf334, (1280, ), (1, ), 0), reinterpret_tensor(buf327, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf329, (320, ), (1, ), 0), buf321, buf323, reinterpret_tensor(buf314, (320, 320), (320, 1), 0), reinterpret_tensor(buf316, (320, ), (1, ), 0), buf312, buf309, buf304, buf306, reinterpret_tensor(buf298, (640, 320), (320, 1), 0), reinterpret_tensor(buf300, (640, ), (1, ), 0), reinterpret_tensor(buf287, (320, 320), (320, 1), 0), reinterpret_tensor(buf289, (320, ), (1, ), 0), buf281, buf283, reinterpret_tensor(buf275, (1280, 320), (320, 1), 0), reinterpret_tensor(buf277, (1280, ), (1, ), 0), reinterpret_tensor(buf270, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf272, (320, ), (1, ), 0), buf264, buf266, reinterpret_tensor(buf257, (320, 320), (320, 1), 0), reinterpret_tensor(buf259, (320, ), (1, ), 0), buf255, buf252, buf247, buf249, reinterpret_tensor(buf241, (640, 320), (320, 1), 0), reinterpret_tensor(buf243, (640, ), (1, ), 0), reinterpret_tensor(buf230, (320, 320), (320, 1), 0), reinterpret_tensor(buf232, (320, ), (1, ), 0), buf224, buf226, reinterpret_tensor(buf218, (1280, 320), (320, 1), 0), reinterpret_tensor(buf220, (1280, ), (1, ), 0), reinterpret_tensor(buf213, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf215, (320, ), (1, ), 0), buf207, buf209, reinterpret_tensor(buf200, (320, 320), (320, 1), 0), reinterpret_tensor(buf202, (320, ), (1, ), 0), buf198, buf195, buf190, buf192, reinterpret_tensor(buf184, (640, 320), (320, 1), 0), reinterpret_tensor(buf186, (640, ), (1, ), 0), reinterpret_tensor(buf173, (320, 320), (320, 1), 0), reinterpret_tensor(buf175, (320, ), (1, ), 0), buf167, buf169, reinterpret_tensor(buf161, (1280, 320), (320, 1), 0), reinterpret_tensor(buf163, (1280, ), (1, ), 0), reinterpret_tensor(buf156, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf158, (320, ), (1, ), 0), buf153, buf150, buf145, buf147, buf138, buf140, reinterpret_tensor(buf132, (512, 512), (512, 1), 0), reinterpret_tensor(buf134, (512, ), (1, ), 0), reinterpret_tensor(buf128, (1024, 512), (512, 1), 0), reinterpret_tensor(buf130, (1024, ), (1, ), 0), reinterpret_tensor(buf117, (512, 512), (512, 1), 0), reinterpret_tensor(buf119, (512, ), (1, ), 0), buf112, buf114, reinterpret_tensor(buf106, (2048, 512), (512, 1), 0), reinterpret_tensor(buf108, (2048, ), (1, ), 0), reinterpret_tensor(buf101, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf103, (512, ), (1, ), 0), buf97, buf94, buf89, buf91, reinterpret_tensor(buf83, (512, 512), (512, 1), 0), reinterpret_tensor(buf85, (512, ), (1, ), 0), reinterpret_tensor(buf79, (1024, 512), (512, 1), 0), reinterpret_tensor(buf81, (1024, ), (1, ), 0), reinterpret_tensor(buf68, (512, 512), (512, 1), 0), reinterpret_tensor(buf70, (512, ), (1, ), 0), buf63, buf65, reinterpret_tensor(buf57, (2048, 512), (512, 1), 0), reinterpret_tensor(buf59, (2048, ), (1, ), 0), reinterpret_tensor(buf52, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf54, (512, ), (1, ), 0), buf47, buf49, reinterpret_tensor(buf41, (512, 512), (512, 1), 0), reinterpret_tensor(buf43, (512, ), (1, ), 0), reinterpret_tensor(buf37, (1024, 512), (512, 1), 0), reinterpret_tensor(buf39, (1024, ), (1, ), 0), reinterpret_tensor(buf26, (512, 512), (512, 1), 0), reinterpret_tensor(buf28, (512, ), (1, ), 0), buf21, buf23, reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(buf17, (2048, ), (1, ), 0), reinterpret_tensor(buf10, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf12, (512, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 512), (512, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 8, 8), (4096, 1, 512, 64), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, 128, 4, 4), (2048, 1, 512, 128), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((320, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_2 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_4 = rand_strided((8, 49, 64), (3136, 64, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((392, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((8, 1, 3136), (3136, 3136, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_10 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_6 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_15 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_13 = rand_strided((8, 49, 64), (3136, 64, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((392, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((8, 1, 3136), (3136, 3136, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_23 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_28 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((8, 49, 64), (3136, 64, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((392, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 1, 49, 64), (6272, 0, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((8, 1, 3136), (3136, 3136, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_35 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_43 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((25088, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((8, 49, 128), (6272, 128, 1), device='cuda:0', dtype=torch.float32)
    view_56 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_44 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 2, 800), (1600, 800, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_49 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_60 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_50 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 49, 128), (6272, 128, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_56 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 2, 800), (1600, 800, 1), device='cuda:0', dtype=torch.float32)
    getitem_60 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_61 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_78 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_80 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 49, 128), (6272, 128, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((8, 2, 800), (1600, 800, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_93 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_53 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 49, 128), (6272, 128, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 2, 49, 64), (12544, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 2, 800), (1600, 800, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_108 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_62 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((6272, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_33 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_71 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_98 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_125 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_143 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_43 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_87 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_122 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_158 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_160 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_48 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_98 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_130 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_133 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_173 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_175 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_53 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_105 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_145 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_146 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_188 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_109 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_114 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_197 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_116 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_199 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_154 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_158 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_203 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_118 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_63 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_207 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_123 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_209 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_166 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_218 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_127 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_68 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_222 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_132 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_227 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_134 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_229 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_178 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_179 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_181 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_233 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_235 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_73 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_237 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_239 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_143 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_190 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_193 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_248 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_78 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_252 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_150 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_257 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_259 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_202 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_203 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_263 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_265 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_83 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_267 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_159 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_269 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_272 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_161 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_214 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_215 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_217 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_278 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_163 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_88 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_284 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_287 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_170 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_289 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_226 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_227 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_229 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_293 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_172 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_295 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_93 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_297 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_177 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_299 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_302 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_179 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_304 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_238 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_239 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_241 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_308 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_181 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_98 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_312 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_186 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_314 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_215 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_317 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_188 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_319 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_250 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_251 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_253 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_323 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_190 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_325 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_103 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_327 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_195 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_332 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_334 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_262 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_265 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_338 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_199 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_340 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_108 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_344 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_347 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_206 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_349 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_274 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_275 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_277 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_353 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_208 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_355 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_113 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_357 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_213 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_359 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_215 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_364 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_286 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_287 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_368 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_217 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_370 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_118 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_372 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_222 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_374 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    view_377 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_224 = rand_strided((8, 49, 320), (15680, 320, 1), device='cuda:0', dtype=torch.float32)
    view_379 = rand_strided((392, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    getitem_298 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_299 = rand_strided((8, 5, 49, 64), (31360, 64, 640, 1), device='cuda:0', dtype=torch.float32)
    getitem_301 = rand_strided((8, 5, 224), (1120, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_383 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_226 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_385 = rand_strided((1568, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_123 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_387 = rand_strided((1568, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_231 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    mul_233 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_391 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    getitem_310 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_311 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_313 = rand_strided((8, 8, 64), (512, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_398 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_235 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_400 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_128 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_402 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_404 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    mul_240 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_407 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    getitem_320 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_321 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_323 = rand_strided((8, 8, 64), (512, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_325 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_414 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_242 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_416 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_133 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_418 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_247 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_420 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    getitem_330 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_331 = rand_strided((8, 8, 49, 64), (50176, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_333 = rand_strided((8, 8, 64), (512, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_334 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_427 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_249 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_429 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_138 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_431 = rand_strided((392, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_254 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    clone_166 = rand_strided((8, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((8, 8, 49, 64), (25088, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_392 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_479 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_542 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_546 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_560 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_575 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_592 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_604 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_617 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_629 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_635 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_646 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_650 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_660 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_671 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_679 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_692 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_696 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_704 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_44 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_710 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_725 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_735 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_742 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_746 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_754 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_760 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_767 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_771 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_775 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_779 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_785 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_792 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_798 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_802 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_806 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    alias_48 = rand_strided((8, 5, 196, 64), (62720, 1, 320, 5), device='cuda:0', dtype=torch.float32)
    permute_812 = rand_strided((640, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_819 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_825 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_829 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_64 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_833 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_49 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    permute_839 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_65 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_846 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_850 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_854 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_858 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_50 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    permute_864 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_68 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_871 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_69 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_875 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_879 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_70 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_883 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_51 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    permute_889 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_896 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_902 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_906 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_910 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_52 = rand_strided((8, 2, 784, 64), (100352, 1, 128, 2), device='cuda:0', dtype=torch.float32)
    permute_916 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_923 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_929 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_933 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_937 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    alias_53 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_943 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_950 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_954 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_958 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_962 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    alias_54 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_968 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_975 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_981 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_985 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_989 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    alias_55 = rand_strided((8, 1, 3136, 64), (200704, 64, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_995 = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1002 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_9, primals_11, primals_17, primals_23, primals_25, primals_29, primals_31, primals_37, primals_43, primals_47, primals_49, primals_55, primals_61, primals_63, primals_65, primals_69, primals_71, primals_77, primals_83, primals_85, primals_89, primals_91, primals_97, primals_103, primals_107, primals_109, primals_115, primals_121, primals_125, primals_127, primals_133, primals_139, primals_141, primals_143, primals_147, primals_149, primals_155, primals_161, primals_163, primals_167, primals_169, primals_175, primals_181, primals_185, primals_187, primals_193, primals_199, primals_203, primals_205, primals_211, primals_217, primals_221, primals_223, primals_229, primals_235, primals_239, primals_241, primals_247, primals_253, primals_257, primals_259, primals_265, primals_271, primals_275, primals_277, primals_283, primals_289, primals_293, primals_295, primals_301, primals_307, primals_311, primals_313, primals_319, primals_325, primals_329, primals_331, primals_337, primals_343, primals_347, primals_349, primals_355, primals_361, primals_365, primals_367, primals_373, primals_379, primals_383, primals_385, primals_391, primals_397, primals_401, primals_403, primals_409, primals_415, primals_419, primals_421, primals_427, primals_433, primals_437, primals_439, primals_445, primals_451, primals_455, primals_457, primals_463, primals_469, primals_471, primals_473, primals_481, primals_487, primals_489, primals_497, primals_503, primals_511, primals_517, primals_521, mul, mul_2, view_1, permute_2, view_4, mul_4, view_6, getitem_6, getitem_7, getitem_9, getitem_10, getitem_11, view_10, mul_6, view_12, addmm_3, view_14, view_16, mul_11, view_19, permute_15, view_22, mul_13, view_24, getitem_18, getitem_19, getitem_21, getitem_22, getitem_23, view_28, mul_15, view_30, addmm_8, view_32, mul_20, view_34, permute_25, view_37, mul_22, view_39, getitem_30, getitem_31, getitem_33, getitem_34, getitem_35, view_43, mul_24, view_45, addmm_13, view_47, permute_34, mul_29, mul_31, view_51, permute_37, view_54, mul_33, view_56, getitem_44, getitem_45, getitem_47, getitem_48, getitem_49, view_60, mul_35, view_62, addmm_18, view_64, view_66, mul_40, view_69, permute_50, view_72, mul_42, view_74, getitem_56, getitem_57, getitem_59, getitem_60, getitem_61, view_78, mul_44, view_80, addmm_23, view_82, mul_49, view_84, permute_60, view_87, mul_51, view_89, getitem_68, getitem_69, getitem_71, getitem_72, getitem_73, view_93, mul_53, view_95, addmm_28, view_97, mul_58, view_99, permute_70, view_102, mul_60, view_104, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_108, mul_62, view_110, addmm_33, view_112, permute_79, mul_67, mul_69, view_116, permute_82, view_119, mul_71, view_121, getitem_94, getitem_95, getitem_97, getitem_98, getitem_99, view_125, mul_73, view_127, addmm_38, view_129, view_131, mul_78, view_134, permute_95, view_137, mul_80, view_139, getitem_106, getitem_107, getitem_109, getitem_110, getitem_111, view_143, mul_82, view_145, addmm_43, view_147, mul_87, view_149, permute_105, view_152, mul_89, view_154, getitem_118, getitem_119, getitem_121, getitem_122, getitem_123, view_158, mul_91, view_160, addmm_48, view_162, mul_96, view_164, permute_115, view_167, mul_98, view_169, getitem_130, getitem_131, getitem_133, getitem_134, getitem_135, view_173, mul_100, view_175, addmm_53, view_177, mul_105, view_179, permute_125, view_182, mul_107, view_184, getitem_142, getitem_143, getitem_145, getitem_146, getitem_147, view_188, mul_109, view_190, addmm_58, view_192, mul_114, view_194, permute_135, view_197, mul_116, view_199, getitem_154, getitem_155, getitem_157, getitem_158, getitem_159, view_203, mul_118, view_205, addmm_63, view_207, mul_123, view_209, permute_145, view_212, mul_125, view_214, getitem_166, getitem_167, getitem_169, getitem_170, getitem_171, view_218, mul_127, view_220, addmm_68, view_222, mul_132, view_224, permute_155, view_227, mul_134, view_229, getitem_178, getitem_179, getitem_181, getitem_182, getitem_183, view_233, mul_136, view_235, addmm_73, view_237, mul_141, view_239, permute_165, view_242, mul_143, view_244, getitem_190, getitem_191, getitem_193, getitem_194, getitem_195, view_248, mul_145, view_250, addmm_78, view_252, mul_150, view_254, permute_175, view_257, mul_152, view_259, getitem_202, getitem_203, getitem_205, getitem_206, getitem_207, view_263, mul_154, view_265, addmm_83, view_267, mul_159, view_269, permute_185, view_272, mul_161, view_274, getitem_214, getitem_215, getitem_217, getitem_218, getitem_219, view_278, mul_163, view_280, addmm_88, view_282, mul_168, view_284, permute_195, view_287, mul_170, view_289, getitem_226, getitem_227, getitem_229, getitem_230, getitem_231, view_293, mul_172, view_295, addmm_93, view_297, mul_177, view_299, permute_205, view_302, mul_179, view_304, getitem_238, getitem_239, getitem_241, getitem_242, getitem_243, view_308, mul_181, view_310, addmm_98, view_312, mul_186, view_314, permute_215, view_317, mul_188, view_319, getitem_250, getitem_251, getitem_253, getitem_254, getitem_255, view_323, mul_190, view_325, addmm_103, view_327, mul_195, view_329, permute_225, view_332, mul_197, view_334, getitem_262, getitem_263, getitem_265, getitem_266, getitem_267, view_338, mul_199, view_340, addmm_108, view_342, mul_204, view_344, permute_235, view_347, mul_206, view_349, getitem_274, getitem_275, getitem_277, getitem_278, getitem_279, view_353, mul_208, view_355, addmm_113, view_357, mul_213, view_359, permute_245, view_362, mul_215, view_364, getitem_286, getitem_287, getitem_289, getitem_290, getitem_291, view_368, mul_217, view_370, addmm_118, view_372, mul_222, view_374, permute_255, view_377, mul_224, view_379, getitem_298, getitem_299, getitem_301, getitem_302, getitem_303, view_383, mul_226, view_385, addmm_123, view_387, permute_264, mul_231, mul_233, view_391, permute_267, getitem_310, getitem_311, getitem_313, getitem_314, getitem_315, view_398, mul_235, view_400, addmm_128, view_402, view_404, mul_240, view_407, permute_278, getitem_320, getitem_321, getitem_323, getitem_324, getitem_325, view_414, mul_242, view_416, addmm_133, view_418, mul_247, view_420, permute_286, getitem_330, getitem_331, getitem_333, getitem_334, getitem_335, view_427, mul_249, view_429, addmm_138, view_431, mul_254, clone_166, permute_294, div_1, permute_298, permute_302, div_2, permute_306, alias_28, permute_312, permute_317, div_3, permute_321, permute_325, div_4, permute_329, alias_29, permute_335, permute_340, div_5, permute_346, permute_350, div_6, permute_354, alias_30, permute_360, permute_365, div_7, div_8, permute_371, permute_375, div_9, permute_379, alias_31, permute_385, div_10, permute_392, div_11, permute_396, permute_400, div_12, permute_404, alias_32, permute_410, div_13, permute_417, div_14, permute_421, permute_425, div_15, permute_429, alias_33, permute_435, div_16, permute_442, div_17, permute_446, permute_450, div_18, permute_454, alias_34, permute_460, div_19, permute_467, div_20, permute_471, permute_475, div_21, permute_479, alias_35, permute_485, div_22, permute_492, div_23, permute_496, permute_500, div_24, permute_504, alias_36, permute_510, div_25, permute_517, div_26, permute_521, permute_525, div_27, permute_529, alias_37, permute_535, div_28, permute_542, div_29, permute_546, permute_550, div_30, permute_554, alias_38, permute_560, div_31, permute_567, div_32, permute_571, permute_575, div_33, permute_579, alias_39, permute_585, div_34, permute_592, div_35, permute_596, permute_600, div_36, permute_604, alias_40, permute_610, div_37, permute_617, div_38, permute_621, permute_625, div_39, permute_629, alias_41, permute_635, div_40, permute_642, div_41, permute_646, permute_650, div_42, permute_654, alias_42, permute_660, div_43, permute_667, div_44, permute_671, permute_675, div_45, permute_679, alias_43, permute_685, div_46, permute_692, div_47, permute_696, permute_700, div_48, permute_704, alias_44, permute_710, div_49, permute_717, div_50, permute_721, permute_725, div_51, permute_729, alias_45, permute_735, div_52, permute_742, div_53, permute_746, permute_750, div_54, permute_754, alias_46, permute_760, div_55, permute_767, div_56, permute_771, permute_775, div_57, permute_779, alias_47, permute_785, div_58, permute_792, div_59, permute_798, permute_802, div_60, permute_806, alias_48, permute_812, div_61, permute_819, div_62, div_63, permute_825, permute_829, div_64, permute_833, alias_49, permute_839, div_65, permute_846, div_66, permute_850, permute_854, div_67, permute_858, alias_50, permute_864, div_68, permute_871, div_69, permute_875, permute_879, div_70, permute_883, alias_51, permute_889, div_71, permute_896, div_72, permute_902, permute_906, div_73, permute_910, alias_52, permute_916, div_74, permute_923, div_75, div_76, permute_929, permute_933, div_77, permute_937, alias_53, permute_943, div_78, permute_950, div_79, permute_954, permute_958, div_80, permute_962, alias_54, permute_968, div_81, permute_975, div_82, permute_981, permute_985, div_83, permute_989, alias_55, permute_995, div_84, permute_1002, div_85, div_86, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('twins_pcpvt_base', benchmark_compiled_module)
