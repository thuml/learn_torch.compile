
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


# kernel path: /tmp/torchinductor_youkaichao/52/c52pnkrugfklhlrdg2vfnypv7rjlddrbplzgttk3vnwd6myedyev.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/wy/cwywgqokhzsrkakj6emknhtkpagraq2gvbwcx2f6mdmvs5fvbue5.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzrffatfyeoycftr5v6nsadnfpg7ahprlcvzna4adcctrmcprpg.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3qq2cevzsglry7esuxcox267mqggoerbp4ndstm4qz5tubmasx.py
# Source Nodes: [hidden_states_155], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_155 => add_113, erf_11, mul_116
triton_poi_fused_gelu_gelu_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
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


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnzdril3hx3ed2bavjv54jkucbv5lbbymuh7kd7yn4m4l3nqhik.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, None)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7hsaprkt2hxmbonifwpumrwxmu5ewhtqjokp4fj6fiwssihs4h.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmshwrnstojtorv7l4n2phxoqzwweenmra4lc3ofesm4j34dm2zp.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbzjtfd23cobxhcvgzssvlhduwfl6lfbd22lurgcuv26alwtp7c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3eoomzwdlia3gz4rzfvhgklnwogg5gwldi74q56vouiz2pz43b2.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crq43ana6ipw4in6tzez7jno3gghbfia4rnmq4eciq4tdq4ehyzj.py
# Source Nodes: [attn_weights_47], Original ATen: [aten._softmax, aten._softmax_backward_data]
# attn_weights_47 => div_17, exp_17, sub_47
triton_per_fused__softmax__softmax_backward_data_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 24576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crrshljsz3r5on3tnyapamtgxdtr2l5whpbzpaefuopnw7ikdiei.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caewr2uh22uo6poafeof3sbjwyyux472qpd4tdx4idu4jxlfqiwi.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskcezppd52eaozbnaun7aw7fr5vk2mprdcmthtyffbxvnmdv73j.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbxdyj7lo2b2zbf6zvf2u7dkc44z3loeeq435j64xapjuswbtuf.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = yindex % 512
    y5 = (yindex // 512)
    y1 = (yindex // 512) % 12
    y2 = (yindex // 6144)
    tmp0 = tl.load(in_ptr0 + (x3 + (64*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (512*x3) + (32768*y5)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3 + (64*y1) + (768*y0) + (393216*y2)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jykcd2scoimonjvmmlldmg634juq3b7xcwxjx7h7wc265hfmhp.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 24576
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2ro7ya2uog5vtqfx2nenxzvstqmr7hwbjdynzq6s43bxd5defw5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5iwcswwyiifinqqoq5a5gvg5s3nefeekrmdtrikjdurx3fadxq.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coy65tf2ukmbgsysc4b7qjg23qjyrdzodpomu4o32qfdxx5yztvj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cyseiwcnnydlbfr342edoizv6vkfzgw5sf7cdqjfijtteimt2kes.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr2, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr8 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr9 + (r1 + (768*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr10 + (r1 + (768*x0)), rmask, other=0.0)
    tmp23 = tl.load(in_ptr11 + (r1 + (768*x0)), rmask, other=0.0)
    tmp25 = tl.load(in_ptr12 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.load(in_ptr13 + (r1 + (768*x0)), rmask, other=0.0)
    tmp37 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp32 = tmp26 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp38 = 768.0
    tmp39 = tmp26 * tmp38
    tmp40 = tmp39 - tmp30
    tmp41 = tmp31 * tmp36
    tmp42 = tmp40 - tmp41
    tmp43 = tmp37 * tmp42
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp24, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp43, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdchllknc2gbo6jk3273fttbn775rsitknkiw4fjj3v3dzjg7htm.py
# Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
# masked_fill_ => full_default_1
triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tl.full([1], False, tl.int1)
    tmp27 = 0.0
    tmp28 = tl.where(tmp26, tmp27, tmp25)
    tmp30 = tl.full([1], 1, tl.int64)
    tmp31 = tmp29 == tmp30
    tmp32 = 1.0
    tmp33 = tmp25 * tmp32
    tmp34 = tl.where(tmp31, tmp27, tmp33)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp28, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp34, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qxy7w2klokgls2xove3643zgszugtde353w7whceel724kq5jj.py
# Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill]
# masked_fill_ => full_default_1
triton_poi_fused_embedding_dense_backward_masked_fill_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_masked_fill_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 787968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qw/cqwpadvdvudandkobfv5qsuok6mlkab22tt7t6mocjjmbkzq44ey.py
# Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill, aten.mul]
# masked_fill_ => full_default_1
triton_poi_fused_embedding_dense_backward_masked_fill_mul_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_masked_fill_mul_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38603520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4keesaiil3xmtjbrzl74rnwqownzuqithnhelknso55zjxfx6pd.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 512)) + (32768*(x0 // 64)) + (393216*(x1 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxf2jvhicp6perfqh6ux45ediesmvlct2jqn7ngv4ttgoukk7xpp.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((512*x1) + (393216*(y0 // 512)) + (y0 % 512)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_103, primals_113, primals_123, primals_129, primals_139, primals_149, primals_155, primals_165, primals_175, primals_181, primals_191, primals_201, primals_207, primals_217, primals_227, primals_233, primals_243, primals_253, primals_259, primals_264, view, add, mul_1, view_1, bmm, amax, sum_1, view_15, mul_4, view_17, addmm_4, view_19, mul_9, view_21, bmm_2, amax_1, sum_2, view_35, mul_12, view_37, addmm_10, view_39, mul_17, view_41, bmm_4, amax_2, sum_3, view_55, mul_20, view_57, addmm_16, view_59, mul_25, view_61, bmm_6, amax_3, sum_4, view_75, mul_28, view_77, addmm_22, view_79, mul_33, view_81, bmm_8, amax_4, sum_5, view_95, mul_36, view_97, addmm_28, view_99, mul_41, view_101, bmm_10, amax_5, sum_6, view_115, mul_44, view_117, addmm_34, view_119, mul_49, mul_52, view_123, view_139, mul_55, view_141, view_143, bmm_14, amax_7, sum_8, view_155, mul_58, view_157, addmm_44, view_159, mul_63, view_161, view_177, mul_66, view_179, bmm_18, amax_9, sum_10, view_193, mul_69, view_195, addmm_54, view_197, mul_74, view_199, view_215, mul_77, view_217, bmm_22, amax_11, sum_12, view_231, mul_80, view_233, addmm_64, view_235, mul_85, view_237, view_253, mul_88, view_255, bmm_26, amax_13, sum_14, view_269, mul_91, view_271, addmm_74, view_273, mul_96, view_275, view_291, mul_99, view_293, bmm_30, amax_15, sum_16, view_307, mul_102, view_309, addmm_84, view_311, mul_107, view_313, view_329, mul_110, view_331, bmm_34, amax_17, sum_18, view_345, mul_113, view_347, addmm_94, view_349, mul_118, view_351, permute_189, div_18, permute_191, permute_195, div_19, permute_199, permute_204, permute_205, permute_206, permute_207, permute_211, permute_216, permute_220, div_20, permute_224, permute_229, permute_230, alias_19, permute_231, permute_232, permute_236, permute_241, permute_245, div_21, permute_249, permute_253, div_22, permute_257, permute_262, permute_263, permute_264, permute_265, permute_269, permute_274, permute_278, div_23, permute_282, permute_287, permute_288, alias_21, permute_289, permute_290, permute_294, permute_299, permute_303, div_24, permute_307, permute_311, div_25, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, div_26, permute_340, permute_345, permute_346, alias_23, permute_347, permute_348, permute_352, permute_357, permute_361, div_27, permute_365, permute_369, div_28, permute_373, permute_378, permute_379, permute_380, permute_381, permute_385, permute_390, permute_394, div_29, permute_398, permute_403, permute_404, alias_25, permute_405, permute_406, permute_410, permute_415, permute_419, div_30, permute_423, permute_427, div_31, permute_431, permute_436, permute_437, permute_438, permute_439, permute_443, permute_448, permute_452, div_32, permute_456, permute_461, permute_462, alias_27, permute_463, permute_464, permute_468, permute_473, permute_477, div_33, permute_481, permute_485, div_34, permute_489, permute_494, permute_495, permute_496, permute_497, permute_501, permute_506, permute_510, div_35, permute_514, permute_519, permute_520, alias_29, permute_521, permute_522, permute_526, permute_531, permute_535, div_36, div_37, permute_539, permute_543, div_38, permute_547, permute_552, permute_553, permute_554, permute_555, permute_559, permute_564, permute_568, div_39, permute_572, permute_576, div_40, permute_580, permute_585, permute_586, permute_587, permute_588, permute_592, permute_597, permute_601, div_41, permute_605, permute_609, div_42, permute_613, permute_618, permute_619, permute_620, permute_621, permute_625, permute_630, permute_634, div_43, permute_638, permute_642, div_44, permute_646, permute_651, permute_652, permute_653, permute_654, permute_658, permute_663, permute_667, div_45, permute_671, permute_675, div_46, permute_679, permute_684, permute_685, permute_686, permute_687, permute_691, permute_696, permute_700, div_47, permute_704, permute_708, div_48, permute_712, permute_717, permute_718, permute_719, permute_720, permute_724, permute_729, permute_733, div_49, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_181, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_264, (4, 512), (512, 1))
    assert_size_stride(view, (4, 512), (512, 1))
    assert_size_stride(add, (4, 512), (512, 1))
    assert_size_stride(mul_1, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_1, (2048, 768), (768, 1))
    assert_size_stride(bmm, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_1, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_15, (2048, 768), (768, 1))
    assert_size_stride(mul_4, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_17, (2048, 768), (768, 1))
    assert_size_stride(addmm_4, (2048, 3072), (3072, 1))
    assert_size_stride(view_19, (2048, 3072), (3072, 1))
    assert_size_stride(mul_9, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_21, (2048, 768), (768, 1))
    assert_size_stride(bmm_2, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_1, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_2, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_35, (2048, 768), (768, 1))
    assert_size_stride(mul_12, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_37, (2048, 768), (768, 1))
    assert_size_stride(addmm_10, (2048, 3072), (3072, 1))
    assert_size_stride(view_39, (2048, 3072), (3072, 1))
    assert_size_stride(mul_17, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_41, (2048, 768), (768, 1))
    assert_size_stride(bmm_4, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_2, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_3, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_55, (2048, 768), (768, 1))
    assert_size_stride(mul_20, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_57, (2048, 768), (768, 1))
    assert_size_stride(addmm_16, (2048, 3072), (3072, 1))
    assert_size_stride(view_59, (2048, 3072), (3072, 1))
    assert_size_stride(mul_25, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_61, (2048, 768), (768, 1))
    assert_size_stride(bmm_6, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_3, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_4, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_75, (2048, 768), (768, 1))
    assert_size_stride(mul_28, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_77, (2048, 768), (768, 1))
    assert_size_stride(addmm_22, (2048, 3072), (3072, 1))
    assert_size_stride(view_79, (2048, 3072), (3072, 1))
    assert_size_stride(mul_33, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_81, (2048, 768), (768, 1))
    assert_size_stride(bmm_8, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_4, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_5, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_95, (2048, 768), (768, 1))
    assert_size_stride(mul_36, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_97, (2048, 768), (768, 1))
    assert_size_stride(addmm_28, (2048, 3072), (3072, 1))
    assert_size_stride(view_99, (2048, 3072), (3072, 1))
    assert_size_stride(mul_41, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_101, (2048, 768), (768, 1))
    assert_size_stride(bmm_10, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_5, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_6, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_115, (2048, 768), (768, 1))
    assert_size_stride(mul_44, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_117, (2048, 768), (768, 1))
    assert_size_stride(addmm_34, (2048, 3072), (3072, 1))
    assert_size_stride(view_119, (2048, 3072), (3072, 1))
    assert_size_stride(mul_49, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_123, (2048, 768), (768, 1))
    assert_size_stride(view_139, (2048, 768), (768, 1))
    assert_size_stride(mul_55, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_141, (2048, 768), (768, 1))
    assert_size_stride(view_143, (2048, 768), (768, 1))
    assert_size_stride(bmm_14, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_7, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_8, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_155, (2048, 768), (768, 1))
    assert_size_stride(mul_58, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_157, (2048, 768), (768, 1))
    assert_size_stride(addmm_44, (2048, 3072), (3072, 1))
    assert_size_stride(view_159, (2048, 3072), (3072, 1))
    assert_size_stride(mul_63, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_161, (2048, 768), (768, 1))
    assert_size_stride(view_177, (2048, 768), (768, 1))
    assert_size_stride(mul_66, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_179, (2048, 768), (768, 1))
    assert_size_stride(bmm_18, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_9, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_10, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_193, (2048, 768), (768, 1))
    assert_size_stride(mul_69, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_195, (2048, 768), (768, 1))
    assert_size_stride(addmm_54, (2048, 3072), (3072, 1))
    assert_size_stride(view_197, (2048, 3072), (3072, 1))
    assert_size_stride(mul_74, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_199, (2048, 768), (768, 1))
    assert_size_stride(view_215, (2048, 768), (768, 1))
    assert_size_stride(mul_77, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_217, (2048, 768), (768, 1))
    assert_size_stride(bmm_22, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_11, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_12, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_231, (2048, 768), (768, 1))
    assert_size_stride(mul_80, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_233, (2048, 768), (768, 1))
    assert_size_stride(addmm_64, (2048, 3072), (3072, 1))
    assert_size_stride(view_235, (2048, 3072), (3072, 1))
    assert_size_stride(mul_85, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_237, (2048, 768), (768, 1))
    assert_size_stride(view_253, (2048, 768), (768, 1))
    assert_size_stride(mul_88, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_255, (2048, 768), (768, 1))
    assert_size_stride(bmm_26, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_13, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_14, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_269, (2048, 768), (768, 1))
    assert_size_stride(mul_91, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_271, (2048, 768), (768, 1))
    assert_size_stride(addmm_74, (2048, 3072), (3072, 1))
    assert_size_stride(view_273, (2048, 3072), (3072, 1))
    assert_size_stride(mul_96, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_275, (2048, 768), (768, 1))
    assert_size_stride(view_291, (2048, 768), (768, 1))
    assert_size_stride(mul_99, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_293, (2048, 768), (768, 1))
    assert_size_stride(bmm_30, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_15, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_16, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_307, (2048, 768), (768, 1))
    assert_size_stride(mul_102, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_309, (2048, 768), (768, 1))
    assert_size_stride(addmm_84, (2048, 3072), (3072, 1))
    assert_size_stride(view_311, (2048, 3072), (3072, 1))
    assert_size_stride(mul_107, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_313, (2048, 768), (768, 1))
    assert_size_stride(view_329, (2048, 768), (768, 1))
    assert_size_stride(mul_110, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_331, (2048, 768), (768, 1))
    assert_size_stride(bmm_34, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(amax_17, (48, 512, 1), (512, 1, 1))
    assert_size_stride(sum_18, (48, 512, 1), (512, 1, 1))
    assert_size_stride(view_345, (2048, 768), (768, 1))
    assert_size_stride(mul_113, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_347, (2048, 768), (768, 1))
    assert_size_stride(addmm_94, (2048, 3072), (3072, 1))
    assert_size_stride(view_349, (2048, 3072), (3072, 1))
    assert_size_stride(mul_118, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_351, (2048, 768), (768, 1))
    assert_size_stride(permute_189, (50265, 768), (768, 1))
    assert_size_stride(div_18, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_191, (768, 3072), (3072, 1))
    assert_size_stride(permute_195, (3072, 768), (768, 1))
    assert_size_stride(div_19, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(permute_204, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_205, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_206, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_207, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_211, (768, 768), (768, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_220, (768, 768), (768, 1))
    assert_size_stride(div_20, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_224, (768, 768), (768, 1))
    assert_size_stride(permute_229, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_230, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_19, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_231, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_232, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_236, (768, 768), (768, 1))
    assert_size_stride(permute_241, (768, 768), (768, 1))
    assert_size_stride(permute_245, (768, 768), (768, 1))
    assert_size_stride(div_21, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 3072), (3072, 1))
    assert_size_stride(permute_253, (3072, 768), (768, 1))
    assert_size_stride(div_22, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_257, (768, 768), (768, 1))
    assert_size_stride(permute_262, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_263, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_264, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_265, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_269, (768, 768), (768, 1))
    assert_size_stride(permute_274, (768, 768), (768, 1))
    assert_size_stride(permute_278, (768, 768), (768, 1))
    assert_size_stride(div_23, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_287, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_288, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_21, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_289, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_290, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_24, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_320, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_321, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_322, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_323, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_26, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 768), (768, 1))
    assert_size_stride(permute_345, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_346, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_23, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_347, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_348, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_352, (768, 768), (768, 1))
    assert_size_stride(permute_357, (768, 768), (768, 1))
    assert_size_stride(permute_361, (768, 768), (768, 1))
    assert_size_stride(div_27, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_365, (768, 3072), (3072, 1))
    assert_size_stride(permute_369, (3072, 768), (768, 1))
    assert_size_stride(div_28, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 768), (768, 1))
    assert_size_stride(permute_378, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_379, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_380, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_381, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_385, (768, 768), (768, 1))
    assert_size_stride(permute_390, (768, 768), (768, 1))
    assert_size_stride(permute_394, (768, 768), (768, 1))
    assert_size_stride(div_29, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_403, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_404, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_25, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_405, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_406, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_410, (768, 768), (768, 1))
    assert_size_stride(permute_415, (768, 768), (768, 1))
    assert_size_stride(permute_419, (768, 768), (768, 1))
    assert_size_stride(div_30, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_423, (768, 3072), (3072, 1))
    assert_size_stride(permute_427, (3072, 768), (768, 1))
    assert_size_stride(div_31, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_436, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_437, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_438, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_439, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_443, (768, 768), (768, 1))
    assert_size_stride(permute_448, (768, 768), (768, 1))
    assert_size_stride(permute_452, (768, 768), (768, 1))
    assert_size_stride(div_32, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_456, (768, 768), (768, 1))
    assert_size_stride(permute_461, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_462, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_27, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_463, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_464, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(permute_473, (768, 768), (768, 1))
    assert_size_stride(permute_477, (768, 768), (768, 1))
    assert_size_stride(div_33, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_481, (768, 3072), (3072, 1))
    assert_size_stride(permute_485, (3072, 768), (768, 1))
    assert_size_stride(div_34, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_489, (768, 768), (768, 1))
    assert_size_stride(permute_494, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_495, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_496, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_497, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(permute_506, (768, 768), (768, 1))
    assert_size_stride(permute_510, (768, 768), (768, 1))
    assert_size_stride(div_35, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_514, (768, 768), (768, 1))
    assert_size_stride(permute_519, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_520, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_29, (48, 512, 512), (262144, 512, 1))
    assert_size_stride(permute_521, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_522, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_526, (768, 768), (768, 1))
    assert_size_stride(permute_531, (768, 768), (768, 1))
    assert_size_stride(permute_535, (768, 768), (768, 1))
    assert_size_stride(div_36, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_37, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_539, (768, 3072), (3072, 1))
    assert_size_stride(permute_543, (3072, 768), (768, 1))
    assert_size_stride(div_38, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_547, (768, 768), (768, 1))
    assert_size_stride(permute_552, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_553, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_554, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_555, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_559, (768, 768), (768, 1))
    assert_size_stride(permute_564, (768, 768), (768, 1))
    assert_size_stride(permute_568, (768, 768), (768, 1))
    assert_size_stride(div_39, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_572, (768, 3072), (3072, 1))
    assert_size_stride(permute_576, (3072, 768), (768, 1))
    assert_size_stride(div_40, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_580, (768, 768), (768, 1))
    assert_size_stride(permute_585, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_586, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_587, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_588, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_592, (768, 768), (768, 1))
    assert_size_stride(permute_597, (768, 768), (768, 1))
    assert_size_stride(permute_601, (768, 768), (768, 1))
    assert_size_stride(div_41, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_605, (768, 3072), (3072, 1))
    assert_size_stride(permute_609, (3072, 768), (768, 1))
    assert_size_stride(div_42, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_613, (768, 768), (768, 1))
    assert_size_stride(permute_618, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_619, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_620, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_621, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_625, (768, 768), (768, 1))
    assert_size_stride(permute_630, (768, 768), (768, 1))
    assert_size_stride(permute_634, (768, 768), (768, 1))
    assert_size_stride(div_43, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_638, (768, 3072), (3072, 1))
    assert_size_stride(permute_642, (3072, 768), (768, 1))
    assert_size_stride(div_44, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_646, (768, 768), (768, 1))
    assert_size_stride(permute_651, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_652, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_653, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_654, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_658, (768, 768), (768, 1))
    assert_size_stride(permute_663, (768, 768), (768, 1))
    assert_size_stride(permute_667, (768, 768), (768, 1))
    assert_size_stride(div_45, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_671, (768, 3072), (3072, 1))
    assert_size_stride(permute_675, (3072, 768), (768, 1))
    assert_size_stride(div_46, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_679, (768, 768), (768, 1))
    assert_size_stride(permute_684, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_685, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_686, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_687, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_691, (768, 768), (768, 1))
    assert_size_stride(permute_696, (768, 768), (768, 1))
    assert_size_stride(permute_700, (768, 768), (768, 1))
    assert_size_stride(div_47, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_704, (768, 3072), (3072, 1))
    assert_size_stride(permute_708, (3072, 768), (768, 1))
    assert_size_stride(div_48, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_712, (768, 768), (768, 1))
    assert_size_stride(permute_717, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_718, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_719, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_720, (48, 512, 64), (32768, 64, 1))
    assert_size_stride(permute_724, (768, 768), (768, 1))
    assert_size_stride(permute_729, (768, 768), (768, 1))
    assert_size_stride(permute_733, (768, 768), (768, 1))
    assert_size_stride(div_49, (4, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (4, 512, 50265), (25735680, 50265, 1))
    assert_size_stride(tangents_2, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_3, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_4, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_5, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_6, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_7, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_8, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_9, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_10, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_11, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_12, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_13, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_14, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_15, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_16, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_17, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_18, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_19, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_20, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_21, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_22, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_23, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_24, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_25, (4, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_26, (4, 512, 768), (393216, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (50265, 2048), (1, 50265), 0), view_351, out=buf0)
        del view_351
        buf1 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (2048, 50265), (50265, 1), 0), permute_189, out=buf1)
        del permute_189
        del tangents_1
        buf4 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_layer_norm_backward_0.run(buf1, primals_259, mul_118, div_18, buf4, 2048, 768, grid=grid(2048), stream=stream0)
        del div_18
        del primals_259
        buf5 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf1, mul_118, buf5, buf7, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_118
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf5, buf6, 768, 16, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf7, buf8, 768, 16, grid=grid(768), stream=stream0)
        buf9 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (2048, 768), (768, 1), 0), permute_191, out=buf9)
        del permute_191
        buf10 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (768, 2048), (1, 768), 0), view_349, out=buf10)
        del view_349
        buf13 = reinterpret_tensor(buf9, (4, 512, 3072), (1572864, 3072, 1), 0); del buf9  # reuse
        # Source Nodes: [hidden_states_155], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf13, addmm_94, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_94
        buf14 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 3072), (3072, 1), 0), permute_195, out=buf14)
        del permute_195
        buf11 = reinterpret_tensor(buf7, (1, 768, 16), (12288, 1, 768), 0); del buf7  # reuse
        buf21 = buf5; del buf5  # reuse
        buf23 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf4, buf14, mul_113, buf11, buf21, buf23, 12288, 128, grid=grid(12288), stream=stream0)
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf11, buf12, 768, 16, grid=grid(768), stream=stream0)
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 2048), (1, 3072), 0), view_347, out=buf15)
        del view_347
        buf16 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf13, buf16, 49152, 128, grid=grid(49152), stream=stream0)
        buf17 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf16, buf17, 3072, 16, grid=grid(3072), stream=stream0)
        buf20 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf4, buf14, primals_253, mul_113, div_19, buf20, 2048, 768, grid=grid(2048), stream=stream0)
        del div_19
        del mul_113
        del primals_253
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf21, buf22, 768, 16, grid=grid(768), stream=stream0)
        buf24 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf23, buf24, 768, 16, grid=grid(768), stream=stream0)
        buf25 = reinterpret_tensor(buf4, (2048, 768), (768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 768), (768, 1), 0), permute_199, out=buf25)
        del permute_199
        buf26 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (768, 2048), (1, 768), 0), view_345, out=buf26)
        del view_345
        buf29 = reinterpret_tensor(buf14, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf25, buf29, 1572864, grid=grid(1572864), stream=stream0)
        buf31 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (48, 512, 64), (32768, 64, 1), 0), permute_205, out=buf31)
        del permute_205
        buf33 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_47], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf31, bmm_34, amax_17, sum_18, buf33, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_17
        del bmm_34
        del sum_18
        buf35 = reinterpret_tensor(buf25, (48, 512, 64), (32768, 64, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf33, permute_207, out=buf35)
        del permute_207
        buf46 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf35, buf46, 1572864, grid=grid(1572864), stream=stream0)
        buf47 = reinterpret_tensor(buf35, (2048, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, permute_220, out=buf47)
        del permute_220
        buf27 = reinterpret_tensor(buf23, (1, 768, 16), (12288, 1, 768), 0); del buf23  # reuse
        buf54 = buf21; del buf21  # reuse
        buf56 = reinterpret_tensor(buf11, (768, 16), (1, 768), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf20, buf47, mul_110, buf27, buf54, buf56, 12288, 128, grid=grid(12288), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf27, buf28, 768, 16, grid=grid(768), stream=stream0)
        buf30 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_204, reinterpret_tensor(buf29, (48, 512, 64), (32768, 64, 1), 0), out=buf30)
        del permute_204
        buf34 = reinterpret_tensor(buf29, (48, 64, 512), (32768, 512, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_206, buf33, out=buf34)
        del permute_206
        buf36 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_25, buf30, buf36, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_25
        buf37 = reinterpret_tensor(buf30, (2048, 768), (768, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2048, 768), (768, 1), 0), permute_211, out=buf37)
        del permute_211
        buf38 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (768, 2048), (1, 768), 0), view_143, out=buf38)
        buf39 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf36, buf39, 12288, 128, grid=grid(12288), stream=stream0)
        buf40 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf39, buf40, 768, 16, grid=grid(768), stream=stream0)
        buf41 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_24, buf34, buf41, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_24
        buf42 = reinterpret_tensor(buf34, (2048, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (2048, 768), (768, 1), 0), permute_216, out=buf42)
        del permute_216
        buf43 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (768, 2048), (1, 768), 0), view_143, out=buf43)
        buf44 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf41, buf44, 12288, 128, grid=grid(12288), stream=stream0)
        buf45 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf44, buf45, 768, 16, grid=grid(768), stream=stream0)
        buf48 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (768, 2048), (1, 768), 0), view_331, out=buf48)
        del view_331
        buf49 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf46, buf49, 12288, 128, grid=grid(12288), stream=stream0)
        buf50 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf49, buf50, 768, 16, grid=grid(768), stream=stream0)
        buf53 = reinterpret_tensor(buf46, (4, 512, 768), (393216, 768, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf20, buf47, primals_243, mul_110, div_20, buf53, 2048, 768, grid=grid(2048), stream=stream0)
        del div_20
        del mul_110
        del primals_243
        buf55 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf54, buf55, 768, 16, grid=grid(768), stream=stream0)
        buf57 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf56, buf57, 768, 16, grid=grid(768), stream=stream0)
        buf58 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (2048, 768), (768, 1), 0), permute_224, out=buf58)
        del permute_224
        buf59 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (768, 2048), (1, 768), 0), view_329, out=buf59)
        del view_329
        buf62 = reinterpret_tensor(buf20, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf58, buf62, 1572864, grid=grid(1572864), stream=stream0)
        buf63 = reinterpret_tensor(buf58, (48, 512, 64), (32768, 64, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_229, reinterpret_tensor(buf62, (48, 512, 64), (32768, 64, 1), 0), out=buf63)
        del permute_229
        buf69 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_23, buf63, buf69, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_23
        buf70 = reinterpret_tensor(buf63, (2048, 768), (768, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (2048, 768), (768, 1), 0), permute_236, out=buf70)
        del permute_236
        buf64 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (48, 512, 64), (32768, 64, 1), 0), permute_230, out=buf64)
        del permute_230
        buf66 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf64, alias_19, buf66, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_19
        buf67 = reinterpret_tensor(buf62, (48, 64, 512), (32768, 512, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_231, reinterpret_tensor(buf66, (48, 512, 512), (262144, 512, 1), 0), out=buf67)
        del permute_231
        buf74 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_22, buf67, buf74, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_22
        buf75 = reinterpret_tensor(buf67, (2048, 768), (768, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (2048, 768), (768, 1), 0), permute_241, out=buf75)
        del permute_241
        buf68 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf66, (48, 512, 512), (262144, 512, 1), 0), permute_232, out=buf68)
        del permute_232
        buf79 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf68, buf79, 1572864, grid=grid(1572864), stream=stream0)
        buf80 = reinterpret_tensor(buf68, (2048, 768), (768, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, permute_245, out=buf80)
        del permute_245
        buf60 = reinterpret_tensor(buf56, (1, 768, 16), (12288, 1, 768), 0); del buf56  # reuse
        buf88 = buf54; del buf54  # reuse
        buf90 = reinterpret_tensor(buf49, (768, 16), (1, 768), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf53, buf70, buf75, buf80, mul_107, buf60, buf88, buf90, 12288, 128, grid=grid(12288), stream=stream0)
        buf61 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf60, buf61, 768, 16, grid=grid(768), stream=stream0)
        buf71 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (768, 2048), (1, 768), 0), view_313, out=buf71)
        buf72 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf69, buf72, 12288, 128, grid=grid(12288), stream=stream0)
        buf73 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf72, buf73, 768, 16, grid=grid(768), stream=stream0)
        buf76 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (768, 2048), (1, 768), 0), view_313, out=buf76)
        buf77 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf74, buf77, 12288, 128, grid=grid(12288), stream=stream0)
        buf78 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf77, buf78, 768, 16, grid=grid(768), stream=stream0)
        buf81 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (768, 2048), (1, 768), 0), view_313, out=buf81)
        del view_313
        buf82 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf79, buf82, 12288, 128, grid=grid(12288), stream=stream0)
        buf83 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf82, buf83, 768, 16, grid=grid(768), stream=stream0)
        buf84 = buf53; del buf53  # reuse
        buf87 = reinterpret_tensor(buf79, (4, 512, 768), (393216, 768, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf84, buf70, buf75, buf80, primals_233, mul_107, div_21, buf87, 2048, 768, grid=grid(2048), stream=stream0)
        del div_21
        del mul_107
        del primals_233
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf88, buf89, 768, 16, grid=grid(768), stream=stream0)
        buf91 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf90, buf91, 768, 16, grid=grid(768), stream=stream0)
        buf92 = reinterpret_tensor(buf13, (2048, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (2048, 768), (768, 1), 0), permute_249, out=buf92)
        del permute_249
        buf93 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (768, 2048), (1, 768), 0), view_311, out=buf93)
        del view_311
        buf96 = reinterpret_tensor(buf92, (4, 512, 3072), (1572864, 3072, 1), 0); del buf92  # reuse
        # Source Nodes: [hidden_states_140], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf96, addmm_84, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_84
        buf97 = reinterpret_tensor(buf84, (2048, 768), (768, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (2048, 3072), (3072, 1), 0), permute_253, out=buf97)
        del permute_253
        buf94 = reinterpret_tensor(buf90, (1, 768, 16), (12288, 1, 768), 0); del buf90  # reuse
        buf104 = buf88; del buf88  # reuse
        buf106 = reinterpret_tensor(buf82, (768, 16), (1, 768), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf87, buf97, mul_102, buf94, buf104, buf106, 12288, 128, grid=grid(12288), stream=stream0)
        buf95 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf94, buf95, 768, 16, grid=grid(768), stream=stream0)
        buf98 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (3072, 2048), (1, 3072), 0), view_309, out=buf98)
        del view_309
        buf99 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf96, buf99, 49152, 128, grid=grid(49152), stream=stream0)
        buf100 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf99, buf100, 3072, 16, grid=grid(3072), stream=stream0)
        buf103 = reinterpret_tensor(buf80, (4, 512, 768), (393216, 768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf87, buf97, primals_227, mul_102, div_22, buf103, 2048, 768, grid=grid(2048), stream=stream0)
        del div_22
        del mul_102
        del primals_227
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf104, buf105, 768, 16, grid=grid(768), stream=stream0)
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf106, buf107, 768, 16, grid=grid(768), stream=stream0)
        buf108 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (2048, 768), (768, 1), 0), permute_257, out=buf108)
        del permute_257
        buf109 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (768, 2048), (1, 768), 0), view_307, out=buf109)
        del view_307
        buf112 = reinterpret_tensor(buf87, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf108, buf112, 1572864, grid=grid(1572864), stream=stream0)
        buf114 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf112, (48, 512, 64), (32768, 64, 1), 0), permute_263, out=buf114)
        del permute_263
        buf116 = buf64; del buf64  # reuse
        # Source Nodes: [attn_weights_41], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf114, bmm_30, amax_15, sum_16, buf116, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_15
        del bmm_30
        del sum_16
        buf118 = reinterpret_tensor(buf108, (48, 512, 64), (32768, 64, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf116, permute_265, out=buf118)
        del permute_265
        buf129 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf118, buf129, 1572864, grid=grid(1572864), stream=stream0)
        buf130 = reinterpret_tensor(buf118, (2048, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf129, permute_278, out=buf130)
        del permute_278
        buf110 = reinterpret_tensor(buf106, (1, 768, 16), (12288, 1, 768), 0); del buf106  # reuse
        buf137 = buf104; del buf104  # reuse
        buf139 = reinterpret_tensor(buf94, (768, 16), (1, 768), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf103, buf130, mul_99, buf110, buf137, buf139, 12288, 128, grid=grid(12288), stream=stream0)
        buf111 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf110, buf111, 768, 16, grid=grid(768), stream=stream0)
        buf113 = reinterpret_tensor(buf70, (48, 512, 64), (32768, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_262, reinterpret_tensor(buf112, (48, 512, 64), (32768, 64, 1), 0), out=buf113)
        del permute_262
        buf117 = reinterpret_tensor(buf112, (48, 64, 512), (32768, 512, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_264, buf116, out=buf117)
        del permute_264
        buf119 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_21, buf113, buf119, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_21
        buf120 = reinterpret_tensor(buf113, (2048, 768), (768, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (2048, 768), (768, 1), 0), permute_269, out=buf120)
        del permute_269
        buf121 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (768, 2048), (1, 768), 0), view_143, out=buf121)
        buf122 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf119, buf122, 12288, 128, grid=grid(12288), stream=stream0)
        buf123 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf122, buf123, 768, 16, grid=grid(768), stream=stream0)
        buf124 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_20, buf117, buf124, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_20
        buf125 = reinterpret_tensor(buf117, (2048, 768), (768, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (2048, 768), (768, 1), 0), permute_274, out=buf125)
        del permute_274
        buf126 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (768, 2048), (1, 768), 0), view_143, out=buf126)
        buf127 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf124, buf127, 12288, 128, grid=grid(12288), stream=stream0)
        buf128 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf127, buf128, 768, 16, grid=grid(768), stream=stream0)
        buf131 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (768, 2048), (1, 768), 0), view_293, out=buf131)
        del view_293
        buf132 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf129, buf132, 12288, 128, grid=grid(12288), stream=stream0)
        buf133 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf132, buf133, 768, 16, grid=grid(768), stream=stream0)
        buf136 = reinterpret_tensor(buf129, (4, 512, 768), (393216, 768, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf103, buf130, primals_217, mul_99, div_23, buf136, 2048, 768, grid=grid(2048), stream=stream0)
        del div_23
        del mul_99
        del primals_217
        buf138 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf137, buf138, 768, 16, grid=grid(768), stream=stream0)
        buf140 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf139, buf140, 768, 16, grid=grid(768), stream=stream0)
        buf141 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (2048, 768), (768, 1), 0), permute_282, out=buf141)
        del permute_282
        buf142 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (768, 2048), (1, 768), 0), view_291, out=buf142)
        del view_291
        buf145 = reinterpret_tensor(buf103, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf141, buf145, 1572864, grid=grid(1572864), stream=stream0)
        buf146 = reinterpret_tensor(buf141, (48, 512, 64), (32768, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_287, reinterpret_tensor(buf145, (48, 512, 64), (32768, 64, 1), 0), out=buf146)
        del permute_287
        buf152 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_19, buf146, buf152, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_19
        buf153 = reinterpret_tensor(buf146, (2048, 768), (768, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (2048, 768), (768, 1), 0), permute_294, out=buf153)
        del permute_294
        buf147 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (48, 512, 64), (32768, 64, 1), 0), permute_288, out=buf147)
        del permute_288
        buf149 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf147, alias_21, buf149, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_21
        buf150 = reinterpret_tensor(buf145, (48, 64, 512), (32768, 512, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_289, reinterpret_tensor(buf149, (48, 512, 512), (262144, 512, 1), 0), out=buf150)
        del permute_289
        buf157 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_18, buf150, buf157, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_18
        buf158 = reinterpret_tensor(buf150, (2048, 768), (768, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 768), (768, 1), 0), permute_299, out=buf158)
        del permute_299
        buf151 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (48, 512, 512), (262144, 512, 1), 0), permute_290, out=buf151)
        del permute_290
        buf162 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf151, buf162, 1572864, grid=grid(1572864), stream=stream0)
        buf163 = reinterpret_tensor(buf151, (2048, 768), (768, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, permute_303, out=buf163)
        del permute_303
        buf143 = reinterpret_tensor(buf139, (1, 768, 16), (12288, 1, 768), 0); del buf139  # reuse
        buf171 = buf137; del buf137  # reuse
        buf173 = reinterpret_tensor(buf132, (768, 16), (1, 768), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf136, buf153, buf158, buf163, mul_96, buf143, buf171, buf173, 12288, 128, grid=grid(12288), stream=stream0)
        buf144 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf143, buf144, 768, 16, grid=grid(768), stream=stream0)
        buf154 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (768, 2048), (1, 768), 0), view_275, out=buf154)
        buf155 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf152, buf155, 12288, 128, grid=grid(12288), stream=stream0)
        buf156 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf155, buf156, 768, 16, grid=grid(768), stream=stream0)
        buf159 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (768, 2048), (1, 768), 0), view_275, out=buf159)
        buf160 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf157, buf160, 12288, 128, grid=grid(12288), stream=stream0)
        buf161 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf160, buf161, 768, 16, grid=grid(768), stream=stream0)
        buf164 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (768, 2048), (1, 768), 0), view_275, out=buf164)
        del view_275
        buf165 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf162, buf165, 12288, 128, grid=grid(12288), stream=stream0)
        buf166 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf165, buf166, 768, 16, grid=grid(768), stream=stream0)
        buf167 = buf136; del buf136  # reuse
        buf170 = reinterpret_tensor(buf162, (4, 512, 768), (393216, 768, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf167, buf153, buf158, buf163, primals_207, mul_96, div_24, buf170, 2048, 768, grid=grid(2048), stream=stream0)
        del div_24
        del mul_96
        del primals_207
        buf172 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf171, buf172, 768, 16, grid=grid(768), stream=stream0)
        buf174 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf173, buf174, 768, 16, grid=grid(768), stream=stream0)
        buf175 = reinterpret_tensor(buf96, (2048, 3072), (3072, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (2048, 768), (768, 1), 0), permute_307, out=buf175)
        del permute_307
        buf176 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (768, 2048), (1, 768), 0), view_273, out=buf176)
        del view_273
        buf179 = reinterpret_tensor(buf175, (4, 512, 3072), (1572864, 3072, 1), 0); del buf175  # reuse
        # Source Nodes: [hidden_states_125], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf179, addmm_74, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_74
        buf180 = reinterpret_tensor(buf167, (2048, 768), (768, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (2048, 3072), (3072, 1), 0), permute_311, out=buf180)
        del permute_311
        buf177 = reinterpret_tensor(buf173, (1, 768, 16), (12288, 1, 768), 0); del buf173  # reuse
        buf187 = buf171; del buf171  # reuse
        buf189 = reinterpret_tensor(buf165, (768, 16), (1, 768), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf170, buf180, mul_91, buf177, buf187, buf189, 12288, 128, grid=grid(12288), stream=stream0)
        buf178 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf177, buf178, 768, 16, grid=grid(768), stream=stream0)
        buf181 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (3072, 2048), (1, 3072), 0), view_271, out=buf181)
        del view_271
        buf182 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf179, buf182, 49152, 128, grid=grid(49152), stream=stream0)
        buf183 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf182, buf183, 3072, 16, grid=grid(3072), stream=stream0)
        buf186 = reinterpret_tensor(buf163, (4, 512, 768), (393216, 768, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf170, buf180, primals_201, mul_91, div_25, buf186, 2048, 768, grid=grid(2048), stream=stream0)
        del div_25
        del mul_91
        del primals_201
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf187, buf188, 768, 16, grid=grid(768), stream=stream0)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf189, buf190, 768, 16, grid=grid(768), stream=stream0)
        buf191 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (2048, 768), (768, 1), 0), permute_315, out=buf191)
        del permute_315
        buf192 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (768, 2048), (1, 768), 0), view_269, out=buf192)
        del view_269
        buf195 = reinterpret_tensor(buf170, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf191, buf195, 1572864, grid=grid(1572864), stream=stream0)
        buf197 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf195, (48, 512, 64), (32768, 64, 1), 0), permute_321, out=buf197)
        del permute_321
        buf199 = buf147; del buf147  # reuse
        # Source Nodes: [attn_weights_35], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf197, bmm_26, amax_13, sum_14, buf199, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_13
        del bmm_26
        del sum_14
        buf201 = reinterpret_tensor(buf191, (48, 512, 64), (32768, 64, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf199, permute_323, out=buf201)
        del permute_323
        buf212 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf201, buf212, 1572864, grid=grid(1572864), stream=stream0)
        buf213 = reinterpret_tensor(buf201, (2048, 768), (768, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf212, permute_336, out=buf213)
        del permute_336
        buf193 = reinterpret_tensor(buf189, (1, 768, 16), (12288, 1, 768), 0); del buf189  # reuse
        buf220 = buf187; del buf187  # reuse
        buf222 = reinterpret_tensor(buf177, (768, 16), (1, 768), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf186, buf213, mul_88, buf193, buf220, buf222, 12288, 128, grid=grid(12288), stream=stream0)
        buf194 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf193, buf194, 768, 16, grid=grid(768), stream=stream0)
        buf196 = reinterpret_tensor(buf153, (48, 512, 64), (32768, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_320, reinterpret_tensor(buf195, (48, 512, 64), (32768, 64, 1), 0), out=buf196)
        del permute_320
        buf200 = reinterpret_tensor(buf195, (48, 64, 512), (32768, 512, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_322, buf199, out=buf200)
        del permute_322
        buf202 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_17, buf196, buf202, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_17
        buf203 = reinterpret_tensor(buf196, (2048, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (2048, 768), (768, 1), 0), permute_327, out=buf203)
        del permute_327
        buf204 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 2048), (1, 768), 0), view_143, out=buf204)
        buf205 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf202, buf205, 12288, 128, grid=grid(12288), stream=stream0)
        buf206 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf205, buf206, 768, 16, grid=grid(768), stream=stream0)
        buf207 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_16, buf200, buf207, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_16
        buf208 = reinterpret_tensor(buf200, (2048, 768), (768, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (2048, 768), (768, 1), 0), permute_332, out=buf208)
        del permute_332
        buf209 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (768, 2048), (1, 768), 0), view_143, out=buf209)
        buf210 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf207, buf210, 12288, 128, grid=grid(12288), stream=stream0)
        buf211 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf210, buf211, 768, 16, grid=grid(768), stream=stream0)
        buf214 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 2048), (1, 768), 0), view_255, out=buf214)
        del view_255
        buf215 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf212, buf215, 12288, 128, grid=grid(12288), stream=stream0)
        buf216 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf215, buf216, 768, 16, grid=grid(768), stream=stream0)
        buf219 = reinterpret_tensor(buf212, (4, 512, 768), (393216, 768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf186, buf213, primals_191, mul_88, div_26, buf219, 2048, 768, grid=grid(2048), stream=stream0)
        del div_26
        del mul_88
        del primals_191
        buf221 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf220, buf221, 768, 16, grid=grid(768), stream=stream0)
        buf223 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf222, buf223, 768, 16, grid=grid(768), stream=stream0)
        buf224 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (2048, 768), (768, 1), 0), permute_340, out=buf224)
        del permute_340
        buf225 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (768, 2048), (1, 768), 0), view_253, out=buf225)
        del view_253
        buf228 = reinterpret_tensor(buf186, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf224, buf228, 1572864, grid=grid(1572864), stream=stream0)
        buf229 = reinterpret_tensor(buf224, (48, 512, 64), (32768, 64, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_345, reinterpret_tensor(buf228, (48, 512, 64), (32768, 64, 1), 0), out=buf229)
        del permute_345
        buf235 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_15, buf229, buf235, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_15
        buf236 = reinterpret_tensor(buf229, (2048, 768), (768, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (2048, 768), (768, 1), 0), permute_352, out=buf236)
        del permute_352
        buf230 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (48, 512, 64), (32768, 64, 1), 0), permute_346, out=buf230)
        del permute_346
        buf232 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf230, alias_23, buf232, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_23
        buf233 = reinterpret_tensor(buf228, (48, 64, 512), (32768, 512, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_347, reinterpret_tensor(buf232, (48, 512, 512), (262144, 512, 1), 0), out=buf233)
        del permute_347
        buf240 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_14, buf233, buf240, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_14
        buf241 = reinterpret_tensor(buf233, (2048, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (2048, 768), (768, 1), 0), permute_357, out=buf241)
        del permute_357
        buf234 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (48, 512, 512), (262144, 512, 1), 0), permute_348, out=buf234)
        del permute_348
        buf245 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf234, buf245, 1572864, grid=grid(1572864), stream=stream0)
        buf246 = reinterpret_tensor(buf234, (2048, 768), (768, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf245, permute_361, out=buf246)
        del permute_361
        buf226 = reinterpret_tensor(buf222, (1, 768, 16), (12288, 1, 768), 0); del buf222  # reuse
        buf254 = buf220; del buf220  # reuse
        buf256 = reinterpret_tensor(buf215, (768, 16), (1, 768), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf219, buf236, buf241, buf246, mul_85, buf226, buf254, buf256, 12288, 128, grid=grid(12288), stream=stream0)
        buf227 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf226, buf227, 768, 16, grid=grid(768), stream=stream0)
        buf237 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (768, 2048), (1, 768), 0), view_237, out=buf237)
        buf238 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf235, buf238, 12288, 128, grid=grid(12288), stream=stream0)
        buf239 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf238, buf239, 768, 16, grid=grid(768), stream=stream0)
        buf242 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (768, 2048), (1, 768), 0), view_237, out=buf242)
        buf243 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf240, buf243, 12288, 128, grid=grid(12288), stream=stream0)
        buf244 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf243, buf244, 768, 16, grid=grid(768), stream=stream0)
        buf247 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (768, 2048), (1, 768), 0), view_237, out=buf247)
        del view_237
        buf248 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf245, buf248, 12288, 128, grid=grid(12288), stream=stream0)
        buf249 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf248, buf249, 768, 16, grid=grid(768), stream=stream0)
        buf250 = buf219; del buf219  # reuse
        buf253 = reinterpret_tensor(buf245, (4, 512, 768), (393216, 768, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf250, buf236, buf241, buf246, primals_181, mul_85, div_27, buf253, 2048, 768, grid=grid(2048), stream=stream0)
        del div_27
        del mul_85
        del primals_181
        buf255 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf254, buf255, 768, 16, grid=grid(768), stream=stream0)
        buf257 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf256, buf257, 768, 16, grid=grid(768), stream=stream0)
        buf258 = reinterpret_tensor(buf179, (2048, 3072), (3072, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (2048, 768), (768, 1), 0), permute_365, out=buf258)
        del permute_365
        buf259 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (768, 2048), (1, 768), 0), view_235, out=buf259)
        del view_235
        buf262 = reinterpret_tensor(buf258, (4, 512, 3072), (1572864, 3072, 1), 0); del buf258  # reuse
        # Source Nodes: [hidden_states_110], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf262, addmm_64, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_64
        buf263 = reinterpret_tensor(buf250, (2048, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (2048, 3072), (3072, 1), 0), permute_369, out=buf263)
        del permute_369
        buf260 = reinterpret_tensor(buf256, (1, 768, 16), (12288, 1, 768), 0); del buf256  # reuse
        buf270 = buf254; del buf254  # reuse
        buf272 = reinterpret_tensor(buf248, (768, 16), (1, 768), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf253, buf263, mul_80, buf260, buf270, buf272, 12288, 128, grid=grid(12288), stream=stream0)
        buf261 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf260, buf261, 768, 16, grid=grid(768), stream=stream0)
        buf264 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (3072, 2048), (1, 3072), 0), view_233, out=buf264)
        del view_233
        buf265 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf262, buf265, 49152, 128, grid=grid(49152), stream=stream0)
        buf266 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf265, buf266, 3072, 16, grid=grid(3072), stream=stream0)
        buf269 = reinterpret_tensor(buf246, (4, 512, 768), (393216, 768, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf253, buf263, primals_175, mul_80, div_28, buf269, 2048, 768, grid=grid(2048), stream=stream0)
        del div_28
        del mul_80
        del primals_175
        buf271 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf270, buf271, 768, 16, grid=grid(768), stream=stream0)
        buf273 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf272, buf273, 768, 16, grid=grid(768), stream=stream0)
        buf274 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (2048, 768), (768, 1), 0), permute_373, out=buf274)
        del permute_373
        buf275 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (768, 2048), (1, 768), 0), view_231, out=buf275)
        del view_231
        buf278 = reinterpret_tensor(buf253, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf274, buf278, 1572864, grid=grid(1572864), stream=stream0)
        buf280 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (48, 512, 64), (32768, 64, 1), 0), permute_379, out=buf280)
        del permute_379
        buf282 = buf230; del buf230  # reuse
        # Source Nodes: [attn_weights_29], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf280, bmm_22, amax_11, sum_12, buf282, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_11
        del bmm_22
        del sum_12
        buf284 = reinterpret_tensor(buf274, (48, 512, 64), (32768, 64, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf282, permute_381, out=buf284)
        del permute_381
        buf296 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf284, buf296, 1572864, grid=grid(1572864), stream=stream0)
        buf297 = reinterpret_tensor(buf284, (2048, 768), (768, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf296, permute_394, out=buf297)
        del permute_394
        buf276 = reinterpret_tensor(buf272, (1, 768, 16), (12288, 1, 768), 0); del buf272  # reuse
        buf304 = buf270; del buf270  # reuse
        buf306 = reinterpret_tensor(buf260, (768, 16), (1, 768), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf269, buf297, mul_77, buf276, buf304, buf306, 12288, 128, grid=grid(12288), stream=stream0)
        buf277 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf276, buf277, 768, 16, grid=grid(768), stream=stream0)
        buf279 = reinterpret_tensor(buf236, (48, 512, 64), (32768, 64, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_378, reinterpret_tensor(buf278, (48, 512, 64), (32768, 64, 1), 0), out=buf279)
        del permute_378
        buf283 = reinterpret_tensor(buf278, (48, 64, 512), (32768, 512, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_380, buf282, out=buf283)
        del permute_380
        buf285 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_13, buf279, buf285, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_13
        buf286 = reinterpret_tensor(buf279, (2048, 768), (768, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (2048, 768), (768, 1), 0), permute_385, out=buf286)
        del permute_385
        buf287 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (768, 2048), (1, 768), 0), view_143, out=buf287)
        buf288 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf285, buf288, 12288, 128, grid=grid(12288), stream=stream0)
        buf289 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf288, buf289, 768, 16, grid=grid(768), stream=stream0)
        buf290 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_12, buf283, buf290, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_12
        buf291 = reinterpret_tensor(buf283, (2048, 768), (768, 1), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (2048, 768), (768, 1), 0), permute_390, out=buf291)
        del permute_390
        buf292 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (768, 2048), (1, 768), 0), view_143, out=buf292)
        buf293 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf290, buf293, 12288, 128, grid=grid(12288), stream=stream0)
        buf294 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf293, buf294, 768, 16, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf290, (4, 512, 768), (393216, 768, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf269, buf297, primals_165, mul_77, div_29, buf303, 2048, 768, grid=grid(2048), stream=stream0)
        del div_29
        del mul_77
        del primals_165
        buf308 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (2048, 768), (768, 1), 0), permute_398, out=buf308)
        del permute_398
        buf312 = reinterpret_tensor(buf269, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf308, buf312, 1572864, grid=grid(1572864), stream=stream0)
        buf313 = reinterpret_tensor(buf308, (48, 512, 64), (32768, 64, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_403, reinterpret_tensor(buf312, (48, 512, 64), (32768, 64, 1), 0), out=buf313)
        del permute_403
        buf319 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_11, buf313, buf319, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_11
        buf320 = reinterpret_tensor(buf313, (2048, 768), (768, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (2048, 768), (768, 1), 0), permute_410, out=buf320)
        del permute_410
        buf314 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (48, 512, 64), (32768, 64, 1), 0), permute_404, out=buf314)
        del permute_404
        buf316 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf314, alias_25, buf316, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_25
        buf317 = reinterpret_tensor(buf312, (48, 64, 512), (32768, 512, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_405, reinterpret_tensor(buf316, (48, 512, 512), (262144, 512, 1), 0), out=buf317)
        del permute_405
        buf324 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_10, buf317, buf324, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_10
        buf325 = reinterpret_tensor(buf317, (2048, 768), (768, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (2048, 768), (768, 1), 0), permute_415, out=buf325)
        del permute_415
        buf318 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (48, 512, 512), (262144, 512, 1), 0), permute_406, out=buf318)
        del permute_406
        buf329 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf318, buf329, 1572864, grid=grid(1572864), stream=stream0)
        buf330 = reinterpret_tensor(buf318, (2048, 768), (768, 1), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf329, permute_419, out=buf330)
        del permute_419
        buf337 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf303, buf320, buf325, buf330, primals_155, mul_74, div_30, buf337, 2048, 768, grid=grid(2048), stream=stream0)
        del div_30
        del primals_155
        buf342 = reinterpret_tensor(buf262, (2048, 3072), (3072, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (2048, 768), (768, 1), 0), permute_423, out=buf342)
        del permute_423
        buf346 = reinterpret_tensor(buf342, (4, 512, 3072), (1572864, 3072, 1), 0); del buf342  # reuse
        # Source Nodes: [hidden_states_95], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf346, addmm_54, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_54
        buf347 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (2048, 3072), (3072, 1), 0), permute_427, out=buf347)
        del permute_427
        buf353 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf337, buf347, primals_149, mul_69, div_31, buf353, 2048, 768, grid=grid(2048), stream=stream0)
        del div_31
        del primals_149
        buf358 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (2048, 768), (768, 1), 0), permute_431, out=buf358)
        del permute_431
        buf362 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf358, buf362, 1572864, grid=grid(1572864), stream=stream0)
        buf363 = reinterpret_tensor(buf358, (48, 512, 64), (32768, 64, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_436, reinterpret_tensor(buf362, (48, 512, 64), (32768, 64, 1), 0), out=buf363)
        del permute_436
        buf369 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_9, buf363, buf369, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_9
        buf370 = reinterpret_tensor(buf363, (2048, 768), (768, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (2048, 768), (768, 1), 0), permute_443, out=buf370)
        del permute_443
        buf364 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf362, (48, 512, 64), (32768, 64, 1), 0), permute_437, out=buf364)
        del permute_437
        buf366 = buf314; del buf314  # reuse
        # Source Nodes: [attn_weights_23], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf364, bmm_18, amax_9, sum_10, buf366, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_9
        del bmm_18
        del sum_10
        buf367 = reinterpret_tensor(buf362, (48, 64, 512), (32768, 512, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_438, buf366, out=buf367)
        del permute_438
        buf374 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_8, buf367, buf374, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_8
        buf375 = reinterpret_tensor(buf367, (2048, 768), (768, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (2048, 768), (768, 1), 0), permute_448, out=buf375)
        del permute_448
        buf368 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf366, permute_439, out=buf368)
        del permute_439
        buf379 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf368, buf379, 1572864, grid=grid(1572864), stream=stream0)
        buf380 = reinterpret_tensor(buf368, (2048, 768), (768, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf379, permute_452, out=buf380)
        del permute_452
        buf386 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf353, buf380, primals_139, mul_66, div_32, buf386, 2048, 768, grid=grid(2048), stream=stream0)
        del div_32
        del primals_139
        buf391 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (2048, 768), (768, 1), 0), permute_456, out=buf391)
        del permute_456
        buf395 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf391, buf395, 1572864, grid=grid(1572864), stream=stream0)
        buf396 = reinterpret_tensor(buf391, (48, 512, 64), (32768, 64, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_461, reinterpret_tensor(buf395, (48, 512, 64), (32768, 64, 1), 0), out=buf396)
        del permute_461
        buf402 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_7, buf396, buf402, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_7
        buf403 = reinterpret_tensor(buf396, (2048, 768), (768, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (2048, 768), (768, 1), 0), permute_468, out=buf403)
        del permute_468
        buf397 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (48, 512, 64), (32768, 64, 1), 0), permute_462, out=buf397)
        del permute_462
        buf399 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf397, alias_27, buf399, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_27
        buf400 = reinterpret_tensor(buf395, (48, 64, 512), (32768, 512, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_463, reinterpret_tensor(buf399, (48, 512, 512), (262144, 512, 1), 0), out=buf400)
        del permute_463
        buf407 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_6, buf400, buf407, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_6
        buf408 = reinterpret_tensor(buf400, (2048, 768), (768, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (2048, 768), (768, 1), 0), permute_473, out=buf408)
        del permute_473
        buf401 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (48, 512, 512), (262144, 512, 1), 0), permute_464, out=buf401)
        del permute_464
        buf412 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf401, buf412, 1572864, grid=grid(1572864), stream=stream0)
        buf413 = reinterpret_tensor(buf401, (2048, 768), (768, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf412, permute_477, out=buf413)
        del permute_477
        buf420 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf386, buf403, buf408, buf413, primals_129, mul_63, div_33, buf420, 2048, 768, grid=grid(2048), stream=stream0)
        del div_33
        del primals_129
        buf425 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (2048, 768), (768, 1), 0), permute_481, out=buf425)
        del permute_481
        buf429 = reinterpret_tensor(buf425, (4, 512, 3072), (1572864, 3072, 1), 0); del buf425  # reuse
        # Source Nodes: [hidden_states_80], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf429, addmm_44, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_44
        buf430 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (2048, 3072), (3072, 1), 0), permute_485, out=buf430)
        del permute_485
        buf436 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf420, buf430, primals_123, mul_58, div_34, buf436, 2048, 768, grid=grid(2048), stream=stream0)
        del div_34
        del primals_123
        buf441 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf436, (2048, 768), (768, 1), 0), permute_489, out=buf441)
        del permute_489
        buf445 = empty((4, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf441, buf445, 1572864, grid=grid(1572864), stream=stream0)
        buf446 = reinterpret_tensor(buf441, (48, 512, 64), (32768, 64, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_494, reinterpret_tensor(buf445, (48, 512, 64), (32768, 64, 1), 0), out=buf446)
        del permute_494
        buf452 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_5, buf446, buf452, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_5
        buf453 = reinterpret_tensor(buf446, (2048, 768), (768, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf452, (2048, 768), (768, 1), 0), permute_501, out=buf453)
        del permute_501
        buf447 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (48, 512, 64), (32768, 64, 1), 0), permute_495, out=buf447)
        del permute_495
        buf449 = buf397; del buf397  # reuse
        # Source Nodes: [attn_weights_17], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf447, bmm_14, amax_7, sum_8, buf449, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_7
        del bmm_14
        del sum_8
        buf450 = reinterpret_tensor(buf445, (48, 64, 512), (32768, 512, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_496, buf449, out=buf450)
        del permute_496
        buf457 = empty((4, 512, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_4, buf450, buf457, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_4
        buf458 = reinterpret_tensor(buf450, (2048, 768), (768, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (2048, 768), (768, 1), 0), permute_506, out=buf458)
        del permute_506
        buf295 = reinterpret_tensor(buf120, (4, 512, 768), (393216, 768, 1), 0); del buf120  # reuse
        buf462 = buf295; del buf295  # reuse
        buf519 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf462, tangents_26, buf37, buf42, buf125, buf203, buf208, buf286, buf291, buf370, buf375, buf453, buf458, primals_100, mul_49, div_37, buf519, 2048, 768, grid=grid(2048), stream=stream0)
        del buf125
        del buf203
        del buf208
        del buf286
        del buf291
        del buf37
        del buf370
        del buf375
        del buf42
        del buf453
        del buf458
        del div_37
        del primals_100
        del tangents_26
        buf298 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (768, 2048), (1, 768), 0), view_217, out=buf298)
        del view_217
        buf299 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf296, buf299, 12288, 128, grid=grid(12288), stream=stream0)
        del buf296
        buf300 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf299, buf300, 768, 16, grid=grid(768), stream=stream0)
        buf305 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf304, buf305, 768, 16, grid=grid(768), stream=stream0)
        buf307 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf306, buf307, 768, 16, grid=grid(768), stream=stream0)
        buf309 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (768, 2048), (1, 768), 0), view_215, out=buf309)
        del view_215
        buf310 = reinterpret_tensor(buf306, (1, 768, 16), (12288, 1, 768), 0); del buf306  # reuse
        buf338 = buf304; del buf304  # reuse
        buf340 = reinterpret_tensor(buf299, (768, 16), (1, 768), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf303, buf320, buf325, buf330, mul_74, buf310, buf338, buf340, 12288, 128, grid=grid(12288), stream=stream0)
        del buf303
        del buf320
        del buf325
        del buf330
        del mul_74
        buf311 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf310, buf311, 768, 16, grid=grid(768), stream=stream0)
        buf321 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (768, 2048), (1, 768), 0), view_199, out=buf321)
        buf322 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf319, buf322, 12288, 128, grid=grid(12288), stream=stream0)
        del buf319
        buf323 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf322, buf323, 768, 16, grid=grid(768), stream=stream0)
        buf326 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (768, 2048), (1, 768), 0), view_199, out=buf326)
        buf327 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf324, buf327, 12288, 128, grid=grid(12288), stream=stream0)
        del buf324
        buf328 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf327, buf328, 768, 16, grid=grid(768), stream=stream0)
        buf331 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (768, 2048), (1, 768), 0), view_199, out=buf331)
        del view_199
        buf332 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf329, buf332, 12288, 128, grid=grid(12288), stream=stream0)
        del buf329
        buf333 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf332, buf333, 768, 16, grid=grid(768), stream=stream0)
        buf339 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf338, buf339, 768, 16, grid=grid(768), stream=stream0)
        buf341 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf340, buf341, 768, 16, grid=grid(768), stream=stream0)
        buf343 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (768, 2048), (1, 768), 0), view_197, out=buf343)
        del view_197
        buf344 = reinterpret_tensor(buf340, (1, 768, 16), (12288, 1, 768), 0); del buf340  # reuse
        buf354 = buf338; del buf338  # reuse
        buf356 = reinterpret_tensor(buf332, (768, 16), (1, 768), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf337, buf347, mul_69, buf344, buf354, buf356, 12288, 128, grid=grid(12288), stream=stream0)
        del buf337
        del buf347
        del mul_69
        buf345 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf344, buf345, 768, 16, grid=grid(768), stream=stream0)
        buf348 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (3072, 2048), (1, 3072), 0), view_195, out=buf348)
        del view_195
        buf349 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf346, buf349, 49152, 128, grid=grid(49152), stream=stream0)
        del buf346
        buf350 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf349, buf350, 3072, 16, grid=grid(3072), stream=stream0)
        buf355 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf354, buf355, 768, 16, grid=grid(768), stream=stream0)
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf356, buf357, 768, 16, grid=grid(768), stream=stream0)
        buf359 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (768, 2048), (1, 768), 0), view_193, out=buf359)
        del view_193
        buf360 = reinterpret_tensor(buf356, (1, 768, 16), (12288, 1, 768), 0); del buf356  # reuse
        buf387 = buf354; del buf354  # reuse
        buf389 = reinterpret_tensor(buf344, (768, 16), (1, 768), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf353, buf380, mul_66, buf360, buf387, buf389, 12288, 128, grid=grid(12288), stream=stream0)
        del buf353
        del buf380
        del mul_66
        buf361 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf360, buf361, 768, 16, grid=grid(768), stream=stream0)
        buf371 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (768, 2048), (1, 768), 0), view_143, out=buf371)
        buf372 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf369, buf372, 12288, 128, grid=grid(12288), stream=stream0)
        del buf369
        buf373 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf372, buf373, 768, 16, grid=grid(768), stream=stream0)
        buf376 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (768, 2048), (1, 768), 0), view_143, out=buf376)
        buf377 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf374, buf377, 12288, 128, grid=grid(12288), stream=stream0)
        del buf374
        buf378 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf377, buf378, 768, 16, grid=grid(768), stream=stream0)
        buf381 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (768, 2048), (1, 768), 0), view_179, out=buf381)
        del view_179
        buf382 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf379, buf382, 12288, 128, grid=grid(12288), stream=stream0)
        del buf379
        buf383 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf382, buf383, 768, 16, grid=grid(768), stream=stream0)
        buf388 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf387, buf388, 768, 16, grid=grid(768), stream=stream0)
        buf390 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf389, buf390, 768, 16, grid=grid(768), stream=stream0)
        buf392 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf386, (768, 2048), (1, 768), 0), view_177, out=buf392)
        del view_177
        buf393 = reinterpret_tensor(buf389, (1, 768, 16), (12288, 1, 768), 0); del buf389  # reuse
        buf421 = buf387; del buf387  # reuse
        buf423 = reinterpret_tensor(buf382, (768, 16), (1, 768), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf386, buf403, buf408, buf413, mul_63, buf393, buf421, buf423, 12288, 128, grid=grid(12288), stream=stream0)
        del buf386
        del buf403
        del buf408
        del buf413
        del mul_63
        buf394 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf393, buf394, 768, 16, grid=grid(768), stream=stream0)
        buf404 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf402, (768, 2048), (1, 768), 0), view_161, out=buf404)
        buf405 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf402, buf405, 12288, 128, grid=grid(12288), stream=stream0)
        del buf402
        buf406 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf405, buf406, 768, 16, grid=grid(768), stream=stream0)
        buf409 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (768, 2048), (1, 768), 0), view_161, out=buf409)
        buf410 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf407, buf410, 12288, 128, grid=grid(12288), stream=stream0)
        buf411 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf410, buf411, 768, 16, grid=grid(768), stream=stream0)
        buf414 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (768, 2048), (1, 768), 0), view_161, out=buf414)
        del view_161
        buf415 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf412, buf415, 12288, 128, grid=grid(12288), stream=stream0)
        buf416 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf415, buf416, 768, 16, grid=grid(768), stream=stream0)
        buf422 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf421, buf422, 768, 16, grid=grid(768), stream=stream0)
        buf424 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf423, buf424, 768, 16, grid=grid(768), stream=stream0)
        buf426 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (768, 2048), (1, 768), 0), view_159, out=buf426)
        del view_159
        buf427 = reinterpret_tensor(buf423, (1, 768, 16), (12288, 1, 768), 0); del buf423  # reuse
        buf437 = buf421; del buf421  # reuse
        buf439 = reinterpret_tensor(buf415, (768, 16), (1, 768), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf420, buf430, mul_58, buf427, buf437, buf439, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_58
        buf428 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf427, buf428, 768, 16, grid=grid(768), stream=stream0)
        buf431 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (3072, 2048), (1, 3072), 0), view_157, out=buf431)
        del view_157
        buf432 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf429, buf432, 49152, 128, grid=grid(49152), stream=stream0)
        buf433 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf432, buf433, 3072, 16, grid=grid(3072), stream=stream0)
        buf438 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf437, buf438, 768, 16, grid=grid(768), stream=stream0)
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf439, buf440, 768, 16, grid=grid(768), stream=stream0)
        buf442 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf436, (768, 2048), (1, 768), 0), view_155, out=buf442)
        del view_155
        buf451 = reinterpret_tensor(buf430, (48, 512, 64), (32768, 64, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf449, permute_497, out=buf451)
        del permute_497
        buf463 = reinterpret_tensor(buf420, (2048, 768), (768, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf451, buf463, 1572864, grid=grid(1572864), stream=stream0)
        buf464 = reinterpret_tensor(buf451, (2048, 768), (768, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf463, permute_510, out=buf464)
        del permute_510
        buf443 = reinterpret_tensor(buf439, (1, 768, 16), (12288, 1, 768), 0); del buf439  # reuse
        buf471 = buf437; del buf437  # reuse
        buf473 = reinterpret_tensor(buf427, (768, 16), (1, 768), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf436, buf464, mul_55, buf443, buf471, buf473, 12288, 128, grid=grid(12288), stream=stream0)
        buf444 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf443, buf444, 768, 16, grid=grid(768), stream=stream0)
        buf454 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf452, (768, 2048), (1, 768), 0), view_143, out=buf454)
        buf455 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf452, buf455, 12288, 128, grid=grid(12288), stream=stream0)
        buf456 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf455, buf456, 768, 16, grid=grid(768), stream=stream0)
        buf459 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (768, 2048), (1, 768), 0), view_143, out=buf459)
        del view_143
        buf460 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf457, buf460, 12288, 128, grid=grid(12288), stream=stream0)
        buf461 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf460, buf461, 768, 16, grid=grid(768), stream=stream0)
        buf465 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (768, 2048), (1, 768), 0), view_141, out=buf465)
        del view_141
        buf466 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf463, buf466, 12288, 128, grid=grid(12288), stream=stream0)
        buf467 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf466, buf467, 768, 16, grid=grid(768), stream=stream0)
        buf470 = reinterpret_tensor(buf463, (4, 512, 768), (393216, 768, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf436, buf464, primals_113, mul_55, div_35, buf470, 2048, 768, grid=grid(2048), stream=stream0)
        del div_35
        del mul_55
        del primals_113
        buf472 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf471, buf472, 768, 16, grid=grid(768), stream=stream0)
        buf474 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf473, buf474, 768, 16, grid=grid(768), stream=stream0)
        buf475 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (2048, 768), (768, 1), 0), permute_514, out=buf475)
        del permute_514
        buf476 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (768, 2048), (1, 768), 0), view_139, out=buf476)
        del view_139
        buf479 = reinterpret_tensor(buf436, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf475, buf479, 1572864, grid=grid(1572864), stream=stream0)
        buf480 = reinterpret_tensor(buf475, (48, 512, 64), (32768, 64, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_519, reinterpret_tensor(buf479, (48, 512, 64), (32768, 64, 1), 0), out=buf480)
        del permute_519
        buf486 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_3, buf480, buf486, 1572864, grid=grid(1572864), stream=stream0)
        del tangents_3
        buf487 = reinterpret_tensor(buf480, (2048, 768), (768, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (2048, 768), (768, 1), 0), permute_526, out=buf487)
        del permute_526
        buf481 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf479, (48, 512, 64), (32768, 64, 1), 0), permute_520, out=buf481)
        del permute_520
        buf483 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf481, alias_29, buf483, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_29
        buf484 = reinterpret_tensor(buf479, (48, 64, 512), (32768, 512, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_521, reinterpret_tensor(buf483, (48, 512, 512), (262144, 512, 1), 0), out=buf484)
        del permute_521
        buf491 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_2, buf484, buf491, 24576, 64, grid=grid(24576, 64), stream=stream0)
        del tangents_2
        buf492 = reinterpret_tensor(buf484, (2048, 768), (768, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (2048, 768), (768, 1), 0), permute_531, out=buf492)
        del permute_531
        buf485 = reinterpret_tensor(buf412, (48, 512, 64), (32768, 64, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf483, (48, 512, 512), (262144, 512, 1), 0), permute_522, out=buf485)
        del permute_522
        buf496 = reinterpret_tensor(buf407, (2048, 768), (768, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf485, buf496, 1572864, grid=grid(1572864), stream=stream0)
        buf497 = reinterpret_tensor(buf485, (2048, 768), (768, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf496, permute_535, out=buf497)
        del permute_535
        buf477 = reinterpret_tensor(buf473, (1, 768, 16), (12288, 1, 768), 0); del buf473  # reuse
        buf505 = buf471; del buf471  # reuse
        buf507 = reinterpret_tensor(buf466, (768, 16), (1, 768), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf470, buf487, buf492, buf497, mul_52, buf477, buf505, buf507, 12288, 128, grid=grid(12288), stream=stream0)
        buf478 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf477, buf478, 768, 16, grid=grid(768), stream=stream0)
        buf488 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (768, 2048), (1, 768), 0), view_123, out=buf488)
        buf489 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf486, buf489, 12288, 128, grid=grid(12288), stream=stream0)
        del buf486
        buf490 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf489, buf490, 768, 16, grid=grid(768), stream=stream0)
        buf493 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (768, 2048), (1, 768), 0), view_123, out=buf493)
        buf494 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf491, buf494, 12288, 128, grid=grid(12288), stream=stream0)
        buf495 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf494, buf495, 768, 16, grid=grid(768), stream=stream0)
        buf498 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (768, 2048), (1, 768), 0), view_123, out=buf498)
        del view_123
        buf499 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf496, buf499, 12288, 128, grid=grid(12288), stream=stream0)
        buf500 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf499, buf500, 768, 16, grid=grid(768), stream=stream0)
        buf501 = buf470; del buf470  # reuse
        buf510 = reinterpret_tensor(buf496, (4, 512, 768), (393216, 768, 1), 0); del buf496  # reuse
        buf514 = reinterpret_tensor(buf491, (4, 512, 768), (393216, 768, 1), 0); del buf491  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_19.run(buf501, buf487, buf492, buf497, primals_103, mul_52, div_36, primals_264, buf510, buf514, 2048, 768, grid=grid(2048), stream=stream0)
        del buf487
        del div_36
        del mul_52
        del primals_103
        buf506 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf505, buf506, 768, 16, grid=grid(768), stream=stream0)
        buf508 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf507, buf508, 768, 16, grid=grid(768), stream=stream0)
        buf509 = empty((1026, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill]
        triton_poi_fused_embedding_dense_backward_masked_fill_20.run(buf509, 787968, grid=grid(787968), stream=stream0)
        aten.index_put_(buf509, [add], buf510, True)
        buf513 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [masked_fill_], Original ATen: [aten.embedding_dense_backward, aten.masked_fill, aten.mul]
        triton_poi_fused_embedding_dense_backward_masked_fill_mul_21.run(buf513, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf513, [primals_264], buf514, True)
        del primals_264
        buf520 = buf507; del buf507  # reuse
        buf522 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf462, mul_49, buf520, buf522, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_49
        buf521 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf520, buf521, 768, 16, grid=grid(768), stream=stream0)
        buf523 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf522, buf523, 768, 16, grid=grid(768), stream=stream0)
        buf524 = reinterpret_tensor(buf429, (2048, 3072), (3072, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf519, (2048, 768), (768, 1), 0), permute_539, out=buf524)
        del permute_539
        buf525 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf519, (768, 2048), (1, 768), 0), view_119, out=buf525)
        del view_119
        buf528 = reinterpret_tensor(buf524, (4, 512, 3072), (1572864, 3072, 1), 0); del buf524  # reuse
        # Source Nodes: [hidden_states_62], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf528, addmm_34, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_34
        buf529 = reinterpret_tensor(buf462, (2048, 768), (768, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf528, (2048, 3072), (3072, 1), 0), permute_543, out=buf529)
        del permute_543
        buf526 = reinterpret_tensor(buf522, (1, 768, 16), (12288, 1, 768), 0); del buf522  # reuse
        buf536 = buf520; del buf520  # reuse
        buf538 = reinterpret_tensor(buf499, (768, 16), (1, 768), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf519, buf529, mul_44, buf526, buf536, buf538, 12288, 128, grid=grid(12288), stream=stream0)
        buf527 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf526, buf527, 768, 16, grid=grid(768), stream=stream0)
        buf530 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf528, (3072, 2048), (1, 3072), 0), view_117, out=buf530)
        del view_117
        buf531 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf528, buf531, 49152, 128, grid=grid(49152), stream=stream0)
        buf532 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf531, buf532, 3072, 16, grid=grid(3072), stream=stream0)
        buf535 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf519, buf529, primals_94, mul_44, div_38, buf535, 2048, 768, grid=grid(2048), stream=stream0)
        del div_38
        del mul_44
        del primals_94
        buf537 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf536, buf537, 768, 16, grid=grid(768), stream=stream0)
        buf539 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf538, buf539, 768, 16, grid=grid(768), stream=stream0)
        buf540 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (2048, 768), (768, 1), 0), permute_547, out=buf540)
        del permute_547
        buf541 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (768, 2048), (1, 768), 0), view_115, out=buf541)
        del view_115
        buf544 = reinterpret_tensor(buf519, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf540, buf544, 1572864, grid=grid(1572864), stream=stream0)
        buf545 = reinterpret_tensor(buf540, (48, 512, 64), (32768, 64, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_552, reinterpret_tensor(buf544, (48, 512, 64), (32768, 64, 1), 0), out=buf545)
        del permute_552
        buf551 = reinterpret_tensor(buf510, (2048, 768), (768, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf545, buf551, 1572864, grid=grid(1572864), stream=stream0)
        buf552 = reinterpret_tensor(buf545, (2048, 768), (768, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf551, permute_559, out=buf552)
        del permute_559
        buf546 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (48, 512, 64), (32768, 64, 1), 0), permute_553, out=buf546)
        del permute_553
        buf548 = buf481; del buf481  # reuse
        # Source Nodes: [attn_weights_11], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf546, bmm_10, amax_5, sum_6, buf548, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_5
        del bmm_10
        del sum_6
        buf549 = reinterpret_tensor(buf544, (48, 64, 512), (32768, 512, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_554, buf548, out=buf549)
        del permute_554
        buf556 = reinterpret_tensor(buf501, (2048, 768), (768, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf549, buf556, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf557 = reinterpret_tensor(buf549, (2048, 768), (768, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf556, permute_564, out=buf557)
        del permute_564
        buf550 = reinterpret_tensor(buf497, (48, 512, 64), (32768, 64, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf548, permute_555, out=buf550)
        del permute_555
        buf561 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf550, buf561, 1572864, grid=grid(1572864), stream=stream0)
        buf562 = reinterpret_tensor(buf550, (2048, 768), (768, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf561, permute_568, out=buf562)
        del permute_568
        buf542 = reinterpret_tensor(buf538, (1, 768, 16), (12288, 1, 768), 0); del buf538  # reuse
        buf570 = buf536; del buf536  # reuse
        buf572 = reinterpret_tensor(buf526, (768, 16), (1, 768), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf535, buf552, buf557, buf562, mul_41, buf542, buf570, buf572, 12288, 128, grid=grid(12288), stream=stream0)
        buf543 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf542, buf543, 768, 16, grid=grid(768), stream=stream0)
        buf553 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (768, 2048), (1, 768), 0), view_101, out=buf553)
        buf554 = buf542; del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf551, buf554, 12288, 128, grid=grid(12288), stream=stream0)
        buf555 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf554, buf555, 768, 16, grid=grid(768), stream=stream0)
        buf558 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf556, (768, 2048), (1, 768), 0), view_101, out=buf558)
        buf559 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf556, buf559, 12288, 128, grid=grid(12288), stream=stream0)
        buf560 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf559, buf560, 768, 16, grid=grid(768), stream=stream0)
        buf563 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (768, 2048), (1, 768), 0), view_101, out=buf563)
        del view_101
        buf564 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf561, buf564, 12288, 128, grid=grid(12288), stream=stream0)
        buf565 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf564, buf565, 768, 16, grid=grid(768), stream=stream0)
        buf566 = buf535; del buf535  # reuse
        buf569 = reinterpret_tensor(buf561, (4, 512, 768), (393216, 768, 1), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf566, buf552, buf557, buf562, primals_84, mul_41, div_39, buf569, 2048, 768, grid=grid(2048), stream=stream0)
        del div_39
        del mul_41
        del primals_84
        buf571 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf570, buf571, 768, 16, grid=grid(768), stream=stream0)
        buf573 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf572, buf573, 768, 16, grid=grid(768), stream=stream0)
        buf574 = reinterpret_tensor(buf528, (2048, 3072), (3072, 1), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (2048, 768), (768, 1), 0), permute_572, out=buf574)
        del permute_572
        buf575 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (768, 2048), (1, 768), 0), view_99, out=buf575)
        del view_99
        buf578 = reinterpret_tensor(buf574, (4, 512, 3072), (1572864, 3072, 1), 0); del buf574  # reuse
        # Source Nodes: [hidden_states_51], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf578, addmm_28, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_28
        buf579 = reinterpret_tensor(buf566, (2048, 768), (768, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (2048, 3072), (3072, 1), 0), permute_576, out=buf579)
        del permute_576
        buf576 = reinterpret_tensor(buf572, (1, 768, 16), (12288, 1, 768), 0); del buf572  # reuse
        buf586 = buf570; del buf570  # reuse
        buf588 = reinterpret_tensor(buf564, (768, 16), (1, 768), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf569, buf579, mul_36, buf576, buf586, buf588, 12288, 128, grid=grid(12288), stream=stream0)
        buf577 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf576, buf577, 768, 16, grid=grid(768), stream=stream0)
        buf580 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf578, (3072, 2048), (1, 3072), 0), view_97, out=buf580)
        del view_97
        buf581 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf578, buf581, 49152, 128, grid=grid(49152), stream=stream0)
        buf582 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf581, buf582, 3072, 16, grid=grid(3072), stream=stream0)
        buf585 = reinterpret_tensor(buf562, (4, 512, 768), (393216, 768, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf569, buf579, primals_78, mul_36, div_40, buf585, 2048, 768, grid=grid(2048), stream=stream0)
        del div_40
        del mul_36
        del primals_78
        buf587 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf586, buf587, 768, 16, grid=grid(768), stream=stream0)
        buf589 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf588, buf589, 768, 16, grid=grid(768), stream=stream0)
        buf590 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (2048, 768), (768, 1), 0), permute_580, out=buf590)
        del permute_580
        buf591 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (768, 2048), (1, 768), 0), view_95, out=buf591)
        del view_95
        buf594 = reinterpret_tensor(buf569, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf590, buf594, 1572864, grid=grid(1572864), stream=stream0)
        buf595 = reinterpret_tensor(buf590, (48, 512, 64), (32768, 64, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_585, reinterpret_tensor(buf594, (48, 512, 64), (32768, 64, 1), 0), out=buf595)
        del permute_585
        buf601 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf595, buf601, 1572864, grid=grid(1572864), stream=stream0)
        buf602 = reinterpret_tensor(buf595, (2048, 768), (768, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf601, permute_592, out=buf602)
        del permute_592
        buf596 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf594, (48, 512, 64), (32768, 64, 1), 0), permute_586, out=buf596)
        del permute_586
        buf598 = buf546; del buf546  # reuse
        # Source Nodes: [attn_weights_9], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf596, bmm_8, amax_4, sum_5, buf598, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_4
        del bmm_8
        del sum_5
        buf599 = reinterpret_tensor(buf594, (48, 64, 512), (32768, 512, 1), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_587, buf598, out=buf599)
        del permute_587
        buf606 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf599, buf606, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf607 = reinterpret_tensor(buf599, (2048, 768), (768, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf606, permute_597, out=buf607)
        del permute_597
        buf600 = reinterpret_tensor(buf556, (48, 512, 64), (32768, 64, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf598, permute_588, out=buf600)
        del permute_588
        buf611 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf600, buf611, 1572864, grid=grid(1572864), stream=stream0)
        buf612 = reinterpret_tensor(buf600, (2048, 768), (768, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf611, permute_601, out=buf612)
        del permute_601
        buf592 = reinterpret_tensor(buf588, (1, 768, 16), (12288, 1, 768), 0); del buf588  # reuse
        buf620 = buf586; del buf586  # reuse
        buf622 = reinterpret_tensor(buf576, (768, 16), (1, 768), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf585, buf602, buf607, buf612, mul_33, buf592, buf620, buf622, 12288, 128, grid=grid(12288), stream=stream0)
        buf593 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf592, buf593, 768, 16, grid=grid(768), stream=stream0)
        buf603 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf601, (768, 2048), (1, 768), 0), view_81, out=buf603)
        buf604 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf601, buf604, 12288, 128, grid=grid(12288), stream=stream0)
        buf605 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf604, buf605, 768, 16, grid=grid(768), stream=stream0)
        buf608 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf606, (768, 2048), (1, 768), 0), view_81, out=buf608)
        buf609 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf606, buf609, 12288, 128, grid=grid(12288), stream=stream0)
        buf610 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf609, buf610, 768, 16, grid=grid(768), stream=stream0)
        buf613 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf611, (768, 2048), (1, 768), 0), view_81, out=buf613)
        del view_81
        buf614 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf611, buf614, 12288, 128, grid=grid(12288), stream=stream0)
        buf615 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf614, buf615, 768, 16, grid=grid(768), stream=stream0)
        buf616 = buf585; del buf585  # reuse
        buf619 = reinterpret_tensor(buf611, (4, 512, 768), (393216, 768, 1), 0); del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf616, buf602, buf607, buf612, primals_68, mul_33, div_41, buf619, 2048, 768, grid=grid(2048), stream=stream0)
        del div_41
        del mul_33
        del primals_68
        buf621 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf620, buf621, 768, 16, grid=grid(768), stream=stream0)
        buf623 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf622, buf623, 768, 16, grid=grid(768), stream=stream0)
        buf624 = reinterpret_tensor(buf578, (2048, 3072), (3072, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (2048, 768), (768, 1), 0), permute_605, out=buf624)
        del permute_605
        buf625 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (768, 2048), (1, 768), 0), view_79, out=buf625)
        del view_79
        buf628 = reinterpret_tensor(buf624, (4, 512, 3072), (1572864, 3072, 1), 0); del buf624  # reuse
        # Source Nodes: [hidden_states_40], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf628, addmm_22, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_22
        buf629 = reinterpret_tensor(buf616, (2048, 768), (768, 1), 0); del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (2048, 3072), (3072, 1), 0), permute_609, out=buf629)
        del permute_609
        buf626 = reinterpret_tensor(buf622, (1, 768, 16), (12288, 1, 768), 0); del buf622  # reuse
        buf636 = buf620; del buf620  # reuse
        buf638 = reinterpret_tensor(buf614, (768, 16), (1, 768), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf619, buf629, mul_28, buf626, buf636, buf638, 12288, 128, grid=grid(12288), stream=stream0)
        buf627 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf626, buf627, 768, 16, grid=grid(768), stream=stream0)
        buf630 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (3072, 2048), (1, 3072), 0), view_77, out=buf630)
        del view_77
        buf631 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf628, buf631, 49152, 128, grid=grid(49152), stream=stream0)
        buf632 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf631, buf632, 3072, 16, grid=grid(3072), stream=stream0)
        buf635 = reinterpret_tensor(buf612, (4, 512, 768), (393216, 768, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf619, buf629, primals_62, mul_28, div_42, buf635, 2048, 768, grid=grid(2048), stream=stream0)
        del div_42
        del mul_28
        del primals_62
        buf637 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf636, buf637, 768, 16, grid=grid(768), stream=stream0)
        buf639 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf638, buf639, 768, 16, grid=grid(768), stream=stream0)
        buf640 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (2048, 768), (768, 1), 0), permute_613, out=buf640)
        del permute_613
        buf641 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (768, 2048), (1, 768), 0), view_75, out=buf641)
        del view_75
        buf644 = reinterpret_tensor(buf619, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf640, buf644, 1572864, grid=grid(1572864), stream=stream0)
        buf645 = reinterpret_tensor(buf640, (48, 512, 64), (32768, 64, 1), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_618, reinterpret_tensor(buf644, (48, 512, 64), (32768, 64, 1), 0), out=buf645)
        del permute_618
        buf651 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf645, buf651, 1572864, grid=grid(1572864), stream=stream0)
        buf652 = reinterpret_tensor(buf645, (2048, 768), (768, 1), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf651, permute_625, out=buf652)
        del permute_625
        buf646 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf644, (48, 512, 64), (32768, 64, 1), 0), permute_619, out=buf646)
        del permute_619
        buf648 = buf596; del buf596  # reuse
        # Source Nodes: [attn_weights_7], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf646, bmm_6, amax_3, sum_4, buf648, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_3
        del bmm_6
        del sum_4
        buf649 = reinterpret_tensor(buf644, (48, 64, 512), (32768, 512, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_620, buf648, out=buf649)
        del permute_620
        buf656 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf649, buf656, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf657 = reinterpret_tensor(buf649, (2048, 768), (768, 1), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf656, permute_630, out=buf657)
        del permute_630
        buf650 = reinterpret_tensor(buf606, (48, 512, 64), (32768, 64, 1), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf648, permute_621, out=buf650)
        del permute_621
        buf661 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf650, buf661, 1572864, grid=grid(1572864), stream=stream0)
        buf662 = reinterpret_tensor(buf650, (2048, 768), (768, 1), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf661, permute_634, out=buf662)
        del permute_634
        buf642 = reinterpret_tensor(buf638, (1, 768, 16), (12288, 1, 768), 0); del buf638  # reuse
        buf670 = buf636; del buf636  # reuse
        buf672 = reinterpret_tensor(buf626, (768, 16), (1, 768), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf635, buf652, buf657, buf662, mul_25, buf642, buf670, buf672, 12288, 128, grid=grid(12288), stream=stream0)
        buf643 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf642, buf643, 768, 16, grid=grid(768), stream=stream0)
        buf653 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf651, (768, 2048), (1, 768), 0), view_61, out=buf653)
        buf654 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf651, buf654, 12288, 128, grid=grid(12288), stream=stream0)
        buf655 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf654, buf655, 768, 16, grid=grid(768), stream=stream0)
        buf658 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (768, 2048), (1, 768), 0), view_61, out=buf658)
        buf659 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf656, buf659, 12288, 128, grid=grid(12288), stream=stream0)
        buf660 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf659, buf660, 768, 16, grid=grid(768), stream=stream0)
        buf663 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf661, (768, 2048), (1, 768), 0), view_61, out=buf663)
        del view_61
        buf664 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf661, buf664, 12288, 128, grid=grid(12288), stream=stream0)
        buf665 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf664, buf665, 768, 16, grid=grid(768), stream=stream0)
        buf666 = buf635; del buf635  # reuse
        buf669 = reinterpret_tensor(buf661, (4, 512, 768), (393216, 768, 1), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf666, buf652, buf657, buf662, primals_52, mul_25, div_43, buf669, 2048, 768, grid=grid(2048), stream=stream0)
        del div_43
        del mul_25
        del primals_52
        buf671 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf670, buf671, 768, 16, grid=grid(768), stream=stream0)
        buf673 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf672, buf673, 768, 16, grid=grid(768), stream=stream0)
        buf674 = reinterpret_tensor(buf628, (2048, 3072), (3072, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (2048, 768), (768, 1), 0), permute_638, out=buf674)
        del permute_638
        buf675 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (768, 2048), (1, 768), 0), view_59, out=buf675)
        del view_59
        buf678 = reinterpret_tensor(buf674, (4, 512, 3072), (1572864, 3072, 1), 0); del buf674  # reuse
        # Source Nodes: [hidden_states_29], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf678, addmm_16, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_16
        buf679 = reinterpret_tensor(buf666, (2048, 768), (768, 1), 0); del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (2048, 3072), (3072, 1), 0), permute_642, out=buf679)
        del permute_642
        buf676 = reinterpret_tensor(buf672, (1, 768, 16), (12288, 1, 768), 0); del buf672  # reuse
        buf686 = buf670; del buf670  # reuse
        buf688 = reinterpret_tensor(buf664, (768, 16), (1, 768), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf669, buf679, mul_20, buf676, buf686, buf688, 12288, 128, grid=grid(12288), stream=stream0)
        buf677 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf676, buf677, 768, 16, grid=grid(768), stream=stream0)
        buf680 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (3072, 2048), (1, 3072), 0), view_57, out=buf680)
        del view_57
        buf681 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf678, buf681, 49152, 128, grid=grid(49152), stream=stream0)
        buf682 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf681, buf682, 3072, 16, grid=grid(3072), stream=stream0)
        buf685 = reinterpret_tensor(buf662, (4, 512, 768), (393216, 768, 1), 0); del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf669, buf679, primals_46, mul_20, div_44, buf685, 2048, 768, grid=grid(2048), stream=stream0)
        del div_44
        del mul_20
        del primals_46
        buf687 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf686, buf687, 768, 16, grid=grid(768), stream=stream0)
        buf689 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf688, buf689, 768, 16, grid=grid(768), stream=stream0)
        buf690 = buf679; del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (2048, 768), (768, 1), 0), permute_646, out=buf690)
        del permute_646
        buf691 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (768, 2048), (1, 768), 0), view_55, out=buf691)
        del view_55
        buf694 = reinterpret_tensor(buf669, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf690, buf694, 1572864, grid=grid(1572864), stream=stream0)
        buf695 = reinterpret_tensor(buf690, (48, 512, 64), (32768, 64, 1), 0); del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_651, reinterpret_tensor(buf694, (48, 512, 64), (32768, 64, 1), 0), out=buf695)
        del permute_651
        buf701 = buf657; del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf695, buf701, 1572864, grid=grid(1572864), stream=stream0)
        buf702 = reinterpret_tensor(buf695, (2048, 768), (768, 1), 0); del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf701, permute_658, out=buf702)
        del permute_658
        buf696 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf694, (48, 512, 64), (32768, 64, 1), 0), permute_652, out=buf696)
        del permute_652
        buf698 = buf646; del buf646  # reuse
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf696, bmm_4, amax_2, sum_3, buf698, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_2
        del bmm_4
        del sum_3
        buf699 = reinterpret_tensor(buf694, (48, 64, 512), (32768, 512, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_653, buf698, out=buf699)
        del permute_653
        buf706 = buf652; del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf699, buf706, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf707 = reinterpret_tensor(buf699, (2048, 768), (768, 1), 0); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf706, permute_663, out=buf707)
        del permute_663
        buf700 = reinterpret_tensor(buf656, (48, 512, 64), (32768, 64, 1), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf698, permute_654, out=buf700)
        del permute_654
        buf711 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf700, buf711, 1572864, grid=grid(1572864), stream=stream0)
        buf712 = reinterpret_tensor(buf700, (2048, 768), (768, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf711, permute_667, out=buf712)
        del permute_667
        buf692 = reinterpret_tensor(buf688, (1, 768, 16), (12288, 1, 768), 0); del buf688  # reuse
        buf720 = buf686; del buf686  # reuse
        buf722 = reinterpret_tensor(buf676, (768, 16), (1, 768), 0); del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf685, buf702, buf707, buf712, mul_17, buf692, buf720, buf722, 12288, 128, grid=grid(12288), stream=stream0)
        buf693 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf692, buf693, 768, 16, grid=grid(768), stream=stream0)
        buf703 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf701, (768, 2048), (1, 768), 0), view_41, out=buf703)
        buf704 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf701, buf704, 12288, 128, grid=grid(12288), stream=stream0)
        buf705 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf704, buf705, 768, 16, grid=grid(768), stream=stream0)
        buf708 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf706, (768, 2048), (1, 768), 0), view_41, out=buf708)
        buf709 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf706, buf709, 12288, 128, grid=grid(12288), stream=stream0)
        buf710 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf709, buf710, 768, 16, grid=grid(768), stream=stream0)
        buf713 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf711, (768, 2048), (1, 768), 0), view_41, out=buf713)
        del view_41
        buf714 = buf709; del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf711, buf714, 12288, 128, grid=grid(12288), stream=stream0)
        buf715 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf714, buf715, 768, 16, grid=grid(768), stream=stream0)
        buf716 = buf685; del buf685  # reuse
        buf719 = reinterpret_tensor(buf711, (4, 512, 768), (393216, 768, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf716, buf702, buf707, buf712, primals_36, mul_17, div_45, buf719, 2048, 768, grid=grid(2048), stream=stream0)
        del div_45
        del mul_17
        del primals_36
        buf721 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf720, buf721, 768, 16, grid=grid(768), stream=stream0)
        buf723 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf722, buf723, 768, 16, grid=grid(768), stream=stream0)
        buf724 = reinterpret_tensor(buf678, (2048, 3072), (3072, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (2048, 768), (768, 1), 0), permute_671, out=buf724)
        del permute_671
        buf725 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (768, 2048), (1, 768), 0), view_39, out=buf725)
        del view_39
        buf728 = reinterpret_tensor(buf724, (4, 512, 3072), (1572864, 3072, 1), 0); del buf724  # reuse
        # Source Nodes: [hidden_states_18], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf728, addmm_10, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_10
        buf729 = reinterpret_tensor(buf716, (2048, 768), (768, 1), 0); del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf728, (2048, 3072), (3072, 1), 0), permute_675, out=buf729)
        del permute_675
        buf726 = reinterpret_tensor(buf722, (1, 768, 16), (12288, 1, 768), 0); del buf722  # reuse
        buf736 = buf720; del buf720  # reuse
        buf738 = reinterpret_tensor(buf714, (768, 16), (1, 768), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf719, buf729, mul_12, buf726, buf736, buf738, 12288, 128, grid=grid(12288), stream=stream0)
        buf727 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf726, buf727, 768, 16, grid=grid(768), stream=stream0)
        buf730 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf728, (3072, 2048), (1, 3072), 0), view_37, out=buf730)
        del view_37
        buf731 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf728, buf731, 49152, 128, grid=grid(49152), stream=stream0)
        buf732 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf731, buf732, 3072, 16, grid=grid(3072), stream=stream0)
        buf735 = reinterpret_tensor(buf712, (4, 512, 768), (393216, 768, 1), 0); del buf712  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf719, buf729, primals_30, mul_12, div_46, buf735, 2048, 768, grid=grid(2048), stream=stream0)
        del div_46
        del mul_12
        del primals_30
        buf737 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf736, buf737, 768, 16, grid=grid(768), stream=stream0)
        buf739 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf738, buf739, 768, 16, grid=grid(768), stream=stream0)
        buf740 = buf729; del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (2048, 768), (768, 1), 0), permute_679, out=buf740)
        del permute_679
        buf741 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf735, (768, 2048), (1, 768), 0), view_35, out=buf741)
        del view_35
        buf744 = reinterpret_tensor(buf719, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf740, buf744, 1572864, grid=grid(1572864), stream=stream0)
        buf745 = reinterpret_tensor(buf740, (48, 512, 64), (32768, 64, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_684, reinterpret_tensor(buf744, (48, 512, 64), (32768, 64, 1), 0), out=buf745)
        del permute_684
        buf751 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf745, buf751, 1572864, grid=grid(1572864), stream=stream0)
        buf752 = reinterpret_tensor(buf745, (2048, 768), (768, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf751, permute_691, out=buf752)
        del permute_691
        buf746 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf744, (48, 512, 64), (32768, 64, 1), 0), permute_685, out=buf746)
        del permute_685
        buf748 = buf696; del buf696  # reuse
        # Source Nodes: [attn_weights_3], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf746, bmm_2, amax_1, sum_2, buf748, 24576, 512, grid=grid(24576), stream=stream0)
        del amax_1
        del bmm_2
        del sum_2
        buf749 = reinterpret_tensor(buf744, (48, 64, 512), (32768, 512, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_686, buf748, out=buf749)
        del permute_686
        buf756 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf749, buf756, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf757 = reinterpret_tensor(buf749, (2048, 768), (768, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf756, permute_696, out=buf757)
        del permute_696
        buf750 = reinterpret_tensor(buf706, (48, 512, 64), (32768, 64, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf748, permute_687, out=buf750)
        del permute_687
        buf761 = buf701; del buf701  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf750, buf761, 1572864, grid=grid(1572864), stream=stream0)
        buf762 = reinterpret_tensor(buf750, (2048, 768), (768, 1), 0); del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf761, permute_700, out=buf762)
        del permute_700
        buf742 = reinterpret_tensor(buf738, (1, 768, 16), (12288, 1, 768), 0); del buf738  # reuse
        buf770 = buf736; del buf736  # reuse
        buf772 = reinterpret_tensor(buf726, (768, 16), (1, 768), 0); del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf735, buf752, buf757, buf762, mul_9, buf742, buf770, buf772, 12288, 128, grid=grid(12288), stream=stream0)
        buf743 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf742, buf743, 768, 16, grid=grid(768), stream=stream0)
        buf753 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf751, (768, 2048), (1, 768), 0), view_21, out=buf753)
        buf754 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf751, buf754, 12288, 128, grid=grid(12288), stream=stream0)
        buf755 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf754, buf755, 768, 16, grid=grid(768), stream=stream0)
        buf758 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (768, 2048), (1, 768), 0), view_21, out=buf758)
        buf759 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf756, buf759, 12288, 128, grid=grid(12288), stream=stream0)
        buf760 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf759, buf760, 768, 16, grid=grid(768), stream=stream0)
        buf763 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (768, 2048), (1, 768), 0), view_21, out=buf763)
        del view_21
        buf764 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf761, buf764, 12288, 128, grid=grid(12288), stream=stream0)
        buf765 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf764, buf765, 768, 16, grid=grid(768), stream=stream0)
        buf766 = buf735; del buf735  # reuse
        buf769 = reinterpret_tensor(buf761, (4, 512, 768), (393216, 768, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf766, buf752, buf757, buf762, primals_20, mul_9, div_47, buf769, 2048, 768, grid=grid(2048), stream=stream0)
        del div_47
        del mul_9
        del primals_20
        buf771 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf770, buf771, 768, 16, grid=grid(768), stream=stream0)
        buf773 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf772, buf773, 768, 16, grid=grid(768), stream=stream0)
        buf774 = reinterpret_tensor(buf728, (2048, 3072), (3072, 1), 0); del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf769, (2048, 768), (768, 1), 0), permute_704, out=buf774)
        del permute_704
        buf775 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf769, (768, 2048), (1, 768), 0), view_19, out=buf775)
        del view_19
        buf778 = reinterpret_tensor(buf774, (4, 512, 3072), (1572864, 3072, 1), 0); del buf774  # reuse
        # Source Nodes: [hidden_states_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf778, addmm_4, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_4
        buf779 = reinterpret_tensor(buf766, (2048, 768), (768, 1), 0); del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (2048, 3072), (3072, 1), 0), permute_708, out=buf779)
        del permute_708
        buf776 = reinterpret_tensor(buf772, (1, 768, 16), (12288, 1, 768), 0); del buf772  # reuse
        buf786 = buf770; del buf770  # reuse
        buf788 = reinterpret_tensor(buf764, (768, 16), (1, 768), 0); del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_4.run(buf769, buf779, mul_4, buf776, buf786, buf788, 12288, 128, grid=grid(12288), stream=stream0)
        buf777 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf776, buf777, 768, 16, grid=grid(768), stream=stream0)
        buf780 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (3072, 2048), (1, 3072), 0), view_17, out=buf780)
        del view_17
        buf781 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf778, buf781, 49152, 128, grid=grid(49152), stream=stream0)
        del buf778
        buf782 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf781, buf782, 3072, 16, grid=grid(3072), stream=stream0)
        del buf781
        buf785 = reinterpret_tensor(buf762, (4, 512, 768), (393216, 768, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf769, buf779, primals_14, mul_4, div_48, buf785, 2048, 768, grid=grid(2048), stream=stream0)
        del div_48
        del mul_4
        del primals_14
        buf787 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf786, buf787, 768, 16, grid=grid(768), stream=stream0)
        buf789 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf788, buf789, 768, 16, grid=grid(768), stream=stream0)
        buf790 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf785, (2048, 768), (768, 1), 0), permute_712, out=buf790)
        del permute_712
        buf791 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf785, (768, 2048), (1, 768), 0), view_15, out=buf791)
        del view_15
        buf794 = reinterpret_tensor(buf769, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf790, buf794, 1572864, grid=grid(1572864), stream=stream0)
        buf795 = reinterpret_tensor(buf790, (48, 512, 64), (32768, 64, 1), 0); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_717, reinterpret_tensor(buf794, (48, 512, 64), (32768, 64, 1), 0), out=buf795)
        del permute_717
        buf801 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_22.run(buf795, buf801, 1572864, grid=grid(1572864), stream=stream0)
        buf802 = reinterpret_tensor(buf795, (2048, 768), (768, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf801, permute_724, out=buf802)
        del permute_724
        buf796 = buf748; del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf794, (48, 512, 64), (32768, 64, 1), 0), permute_718, out=buf796)
        del permute_718
        buf798 = buf746; del buf746  # reuse
        # Source Nodes: [attn_weights_1], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf796, bmm, amax, sum_1, buf798, 24576, 512, grid=grid(24576), stream=stream0)
        del amax
        del bmm
        del buf796
        del sum_1
        buf799 = reinterpret_tensor(buf794, (48, 64, 512), (32768, 512, 1), 0); del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_719, buf798, out=buf799)
        del permute_719
        buf806 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_23.run(buf799, buf806, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf807 = reinterpret_tensor(buf799, (2048, 768), (768, 1), 0); del buf799  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf806, permute_729, out=buf807)
        del permute_729
        buf800 = reinterpret_tensor(buf756, (48, 512, 64), (32768, 64, 1), 0); del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf798, permute_720, out=buf800)
        del buf798
        del permute_720
        buf811 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_10.run(buf800, buf811, 1572864, grid=grid(1572864), stream=stream0)
        buf812 = reinterpret_tensor(buf800, (2048, 768), (768, 1), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf811, permute_733, out=buf812)
        del permute_733
        buf792 = reinterpret_tensor(buf788, (1, 768, 16), (12288, 1, 768), 0); del buf788  # reuse
        buf820 = buf786; del buf786  # reuse
        buf822 = reinterpret_tensor(buf776, (768, 16), (1, 768), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf785, buf802, buf807, buf812, mul_1, buf792, buf820, buf822, 12288, 128, grid=grid(12288), stream=stream0)
        buf793 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf792, buf793, 768, 16, grid=grid(768), stream=stream0)
        buf803 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (768, 2048), (1, 768), 0), view_1, out=buf803)
        buf804 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf801, buf804, 12288, 128, grid=grid(12288), stream=stream0)
        del buf801
        buf805 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf804, buf805, 768, 16, grid=grid(768), stream=stream0)
        buf808 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf806, (768, 2048), (1, 768), 0), view_1, out=buf808)
        buf809 = buf804; del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf806, buf809, 12288, 128, grid=grid(12288), stream=stream0)
        buf810 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf809, buf810, 768, 16, grid=grid(768), stream=stream0)
        buf813 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf811, (768, 2048), (1, 768), 0), view_1, out=buf813)
        del view_1
        buf814 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf811, buf814, 12288, 128, grid=grid(12288), stream=stream0)
        buf815 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf814, buf815, 768, 16, grid=grid(768), stream=stream0)
        del buf814
        buf816 = buf785; del buf785  # reuse
        buf825 = reinterpret_tensor(buf811, (4, 512, 768), (393216, 768, 1), 0); del buf811  # reuse
        buf829 = reinterpret_tensor(buf806, (4, 512, 768), (393216, 768, 1), 0); del buf806  # reuse
        # Source Nodes: [masked_fill_], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_19.run(buf816, buf802, buf807, buf812, primals_4, mul_1, div_49, view, buf825, buf829, 2048, 768, grid=grid(2048), stream=stream0)
        del buf802
        del buf807
        del buf812
        del buf816
        del div_49
        del mul_1
        del primals_4
        buf821 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf820, buf821, 768, 16, grid=grid(768), stream=stream0)
        del buf820
        buf823 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf822, buf823, 768, 16, grid=grid(768), stream=stream0)
        del buf822
        buf824 = empty((1026, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_masked_fill_20.run(buf824, 787968, grid=grid(787968), stream=stream0)
        aten.index_put_(buf824, [add], buf825, True)
        del add
        del buf825
        buf828 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_masked_fill_mul_21.run(buf828, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf828, [view], buf829, True)
        del buf829
        del view
        return (buf824, buf509, buf828, buf821, buf823, reinterpret_tensor(buf813, (768, 768), (768, 1), 0), reinterpret_tensor(buf815, (768, ), (1, ), 0), reinterpret_tensor(buf808, (768, 768), (768, 1), 0), reinterpret_tensor(buf810, (768, ), (1, ), 0), reinterpret_tensor(buf803, (768, 768), (768, 1), 0), reinterpret_tensor(buf805, (768, ), (1, ), 0), reinterpret_tensor(buf791, (768, 768), (768, 1), 0), reinterpret_tensor(buf793, (768, ), (1, ), 0), buf787, buf789, reinterpret_tensor(buf780, (3072, 768), (768, 1), 0), reinterpret_tensor(buf782, (3072, ), (1, ), 0), reinterpret_tensor(buf775, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf777, (768, ), (1, ), 0), buf771, buf773, reinterpret_tensor(buf763, (768, 768), (768, 1), 0), reinterpret_tensor(buf765, (768, ), (1, ), 0), reinterpret_tensor(buf758, (768, 768), (768, 1), 0), reinterpret_tensor(buf760, (768, ), (1, ), 0), reinterpret_tensor(buf753, (768, 768), (768, 1), 0), reinterpret_tensor(buf755, (768, ), (1, ), 0), reinterpret_tensor(buf741, (768, 768), (768, 1), 0), reinterpret_tensor(buf743, (768, ), (1, ), 0), buf737, buf739, reinterpret_tensor(buf730, (3072, 768), (768, 1), 0), reinterpret_tensor(buf732, (3072, ), (1, ), 0), reinterpret_tensor(buf725, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf727, (768, ), (1, ), 0), buf721, buf723, reinterpret_tensor(buf713, (768, 768), (768, 1), 0), reinterpret_tensor(buf715, (768, ), (1, ), 0), reinterpret_tensor(buf708, (768, 768), (768, 1), 0), reinterpret_tensor(buf710, (768, ), (1, ), 0), reinterpret_tensor(buf703, (768, 768), (768, 1), 0), reinterpret_tensor(buf705, (768, ), (1, ), 0), reinterpret_tensor(buf691, (768, 768), (768, 1), 0), reinterpret_tensor(buf693, (768, ), (1, ), 0), buf687, buf689, reinterpret_tensor(buf680, (3072, 768), (768, 1), 0), reinterpret_tensor(buf682, (3072, ), (1, ), 0), reinterpret_tensor(buf675, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf677, (768, ), (1, ), 0), buf671, buf673, reinterpret_tensor(buf663, (768, 768), (768, 1), 0), reinterpret_tensor(buf665, (768, ), (1, ), 0), reinterpret_tensor(buf658, (768, 768), (768, 1), 0), reinterpret_tensor(buf660, (768, ), (1, ), 0), reinterpret_tensor(buf653, (768, 768), (768, 1), 0), reinterpret_tensor(buf655, (768, ), (1, ), 0), reinterpret_tensor(buf641, (768, 768), (768, 1), 0), reinterpret_tensor(buf643, (768, ), (1, ), 0), buf637, buf639, reinterpret_tensor(buf630, (3072, 768), (768, 1), 0), reinterpret_tensor(buf632, (3072, ), (1, ), 0), reinterpret_tensor(buf625, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf627, (768, ), (1, ), 0), buf621, buf623, reinterpret_tensor(buf613, (768, 768), (768, 1), 0), reinterpret_tensor(buf615, (768, ), (1, ), 0), reinterpret_tensor(buf608, (768, 768), (768, 1), 0), reinterpret_tensor(buf610, (768, ), (1, ), 0), reinterpret_tensor(buf603, (768, 768), (768, 1), 0), reinterpret_tensor(buf605, (768, ), (1, ), 0), reinterpret_tensor(buf591, (768, 768), (768, 1), 0), reinterpret_tensor(buf593, (768, ), (1, ), 0), buf587, buf589, reinterpret_tensor(buf580, (3072, 768), (768, 1), 0), reinterpret_tensor(buf582, (3072, ), (1, ), 0), reinterpret_tensor(buf575, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf577, (768, ), (1, ), 0), buf571, buf573, reinterpret_tensor(buf563, (768, 768), (768, 1), 0), reinterpret_tensor(buf565, (768, ), (1, ), 0), reinterpret_tensor(buf558, (768, 768), (768, 1), 0), reinterpret_tensor(buf560, (768, ), (1, ), 0), reinterpret_tensor(buf553, (768, 768), (768, 1), 0), reinterpret_tensor(buf555, (768, ), (1, ), 0), reinterpret_tensor(buf541, (768, 768), (768, 1), 0), reinterpret_tensor(buf543, (768, ), (1, ), 0), buf537, buf539, reinterpret_tensor(buf530, (3072, 768), (768, 1), 0), reinterpret_tensor(buf532, (3072, ), (1, ), 0), reinterpret_tensor(buf525, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf527, (768, ), (1, ), 0), buf521, buf523, buf513, buf506, buf508, reinterpret_tensor(buf498, (768, 768), (768, 1), 0), reinterpret_tensor(buf500, (768, ), (1, ), 0), reinterpret_tensor(buf493, (768, 768), (768, 1), 0), reinterpret_tensor(buf495, (768, ), (1, ), 0), reinterpret_tensor(buf488, (768, 768), (768, 1), 0), reinterpret_tensor(buf490, (768, ), (1, ), 0), reinterpret_tensor(buf476, (768, 768), (768, 1), 0), reinterpret_tensor(buf478, (768, ), (1, ), 0), buf472, buf474, reinterpret_tensor(buf465, (768, 768), (768, 1), 0), reinterpret_tensor(buf467, (768, ), (1, ), 0), reinterpret_tensor(buf459, (768, 768), (768, 1), 0), reinterpret_tensor(buf461, (768, ), (1, ), 0), reinterpret_tensor(buf454, (768, 768), (768, 1), 0), reinterpret_tensor(buf456, (768, ), (1, ), 0), reinterpret_tensor(buf442, (768, 768), (768, 1), 0), reinterpret_tensor(buf444, (768, ), (1, ), 0), buf438, buf440, reinterpret_tensor(buf431, (3072, 768), (768, 1), 0), reinterpret_tensor(buf433, (3072, ), (1, ), 0), reinterpret_tensor(buf426, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf428, (768, ), (1, ), 0), buf422, buf424, reinterpret_tensor(buf414, (768, 768), (768, 1), 0), reinterpret_tensor(buf416, (768, ), (1, ), 0), reinterpret_tensor(buf409, (768, 768), (768, 1), 0), reinterpret_tensor(buf411, (768, ), (1, ), 0), reinterpret_tensor(buf404, (768, 768), (768, 1), 0), reinterpret_tensor(buf406, (768, ), (1, ), 0), reinterpret_tensor(buf392, (768, 768), (768, 1), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), buf388, buf390, reinterpret_tensor(buf381, (768, 768), (768, 1), 0), reinterpret_tensor(buf383, (768, ), (1, ), 0), reinterpret_tensor(buf376, (768, 768), (768, 1), 0), reinterpret_tensor(buf378, (768, ), (1, ), 0), reinterpret_tensor(buf371, (768, 768), (768, 1), 0), reinterpret_tensor(buf373, (768, ), (1, ), 0), reinterpret_tensor(buf359, (768, 768), (768, 1), 0), reinterpret_tensor(buf361, (768, ), (1, ), 0), buf355, buf357, reinterpret_tensor(buf348, (3072, 768), (768, 1), 0), reinterpret_tensor(buf350, (3072, ), (1, ), 0), reinterpret_tensor(buf343, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf345, (768, ), (1, ), 0), buf339, buf341, reinterpret_tensor(buf331, (768, 768), (768, 1), 0), reinterpret_tensor(buf333, (768, ), (1, ), 0), reinterpret_tensor(buf326, (768, 768), (768, 1), 0), reinterpret_tensor(buf328, (768, ), (1, ), 0), reinterpret_tensor(buf321, (768, 768), (768, 1), 0), reinterpret_tensor(buf323, (768, ), (1, ), 0), reinterpret_tensor(buf309, (768, 768), (768, 1), 0), reinterpret_tensor(buf311, (768, ), (1, ), 0), buf305, buf307, reinterpret_tensor(buf298, (768, 768), (768, 1), 0), reinterpret_tensor(buf300, (768, ), (1, ), 0), reinterpret_tensor(buf292, (768, 768), (768, 1), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, 768), (768, 1), 0), reinterpret_tensor(buf289, (768, ), (1, ), 0), reinterpret_tensor(buf275, (768, 768), (768, 1), 0), reinterpret_tensor(buf277, (768, ), (1, ), 0), buf271, buf273, reinterpret_tensor(buf264, (3072, 768), (768, 1), 0), reinterpret_tensor(buf266, (3072, ), (1, ), 0), reinterpret_tensor(buf259, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf261, (768, ), (1, ), 0), buf255, buf257, reinterpret_tensor(buf247, (768, 768), (768, 1), 0), reinterpret_tensor(buf249, (768, ), (1, ), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, ), (1, ), 0), reinterpret_tensor(buf237, (768, 768), (768, 1), 0), reinterpret_tensor(buf239, (768, ), (1, ), 0), reinterpret_tensor(buf225, (768, 768), (768, 1), 0), reinterpret_tensor(buf227, (768, ), (1, ), 0), buf221, buf223, reinterpret_tensor(buf214, (768, 768), (768, 1), 0), reinterpret_tensor(buf216, (768, ), (1, ), 0), reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (768, ), (1, ), 0), reinterpret_tensor(buf204, (768, 768), (768, 1), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, 768), (768, 1), 0), reinterpret_tensor(buf194, (768, ), (1, ), 0), buf188, buf190, reinterpret_tensor(buf181, (3072, 768), (768, 1), 0), reinterpret_tensor(buf183, (3072, ), (1, ), 0), reinterpret_tensor(buf176, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf178, (768, ), (1, ), 0), buf172, buf174, reinterpret_tensor(buf164, (768, 768), (768, 1), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf159, (768, 768), (768, 1), 0), reinterpret_tensor(buf161, (768, ), (1, ), 0), reinterpret_tensor(buf154, (768, 768), (768, 1), 0), reinterpret_tensor(buf156, (768, ), (1, ), 0), reinterpret_tensor(buf142, (768, 768), (768, 1), 0), reinterpret_tensor(buf144, (768, ), (1, ), 0), buf138, buf140, reinterpret_tensor(buf131, (768, 768), (768, 1), 0), reinterpret_tensor(buf133, (768, ), (1, ), 0), reinterpret_tensor(buf126, (768, 768), (768, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf121, (768, 768), (768, 1), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, 768), (768, 1), 0), reinterpret_tensor(buf111, (768, ), (1, ), 0), buf105, buf107, reinterpret_tensor(buf98, (3072, 768), (768, 1), 0), reinterpret_tensor(buf100, (3072, ), (1, ), 0), reinterpret_tensor(buf93, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf95, (768, ), (1, ), 0), buf89, buf91, reinterpret_tensor(buf81, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf76, (768, 768), (768, 1), 0), reinterpret_tensor(buf78, (768, ), (1, ), 0), reinterpret_tensor(buf71, (768, 768), (768, 1), 0), reinterpret_tensor(buf73, (768, ), (1, ), 0), reinterpret_tensor(buf59, (768, 768), (768, 1), 0), reinterpret_tensor(buf61, (768, ), (1, ), 0), buf55, buf57, reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf50, (768, ), (1, ), 0), reinterpret_tensor(buf43, (768, 768), (768, 1), 0), reinterpret_tensor(buf45, (768, ), (1, ), 0), reinterpret_tensor(buf38, (768, 768), (768, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), reinterpret_tensor(buf26, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), buf22, buf24, reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf17, (3072, ), (1, ), 0), reinterpret_tensor(buf10, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), buf6, buf8, reinterpret_tensor(buf0, (50265, 768), (768, 1), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    add = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_4 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_4 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_2 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_6 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_3 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_4 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_8 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_4 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_5 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_10 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_5 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_14 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_7 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_8 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_159 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_18 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_9 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_10 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_193 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_195 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_197 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_74 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_199 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_215 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_217 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_22 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_11 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_12 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_231 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_235 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_237 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_253 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_255 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_26 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_13 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_14 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_269 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_271 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_74 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_273 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_275 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_291 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_99 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_293 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_30 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_15 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_16 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_307 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_309 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_84 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_311 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_107 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_313 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_110 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_331 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    bmm_34 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    amax_17 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_18 = rand_strided((48, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    view_345 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_113 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_347 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_94 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_349 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_118 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_351 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_357 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_378 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_385 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_405 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_461 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_473 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_489 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_495 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((48, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_522 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_531 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_539 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_543 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_553 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_559 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_564 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_568 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_572 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_576 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_580 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_585 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_586 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_588 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_592 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_597 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_605 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_609 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_613 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_618 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_619 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_620 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_630 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_634 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_638 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_642 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_646 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_651 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_652 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_653 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_658 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_663 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_667 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_671 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_679 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_684 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_685 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_686 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_687 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_691 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_696 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_704 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_708 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_712 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_718 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_719 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_720 = rand_strided((48, 512, 64), (32768, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_724 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_733 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 512, 50265), (25735680, 50265, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((4, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_103, primals_113, primals_123, primals_129, primals_139, primals_149, primals_155, primals_165, primals_175, primals_181, primals_191, primals_201, primals_207, primals_217, primals_227, primals_233, primals_243, primals_253, primals_259, primals_264, view, add, mul_1, view_1, bmm, amax, sum_1, view_15, mul_4, view_17, addmm_4, view_19, mul_9, view_21, bmm_2, amax_1, sum_2, view_35, mul_12, view_37, addmm_10, view_39, mul_17, view_41, bmm_4, amax_2, sum_3, view_55, mul_20, view_57, addmm_16, view_59, mul_25, view_61, bmm_6, amax_3, sum_4, view_75, mul_28, view_77, addmm_22, view_79, mul_33, view_81, bmm_8, amax_4, sum_5, view_95, mul_36, view_97, addmm_28, view_99, mul_41, view_101, bmm_10, amax_5, sum_6, view_115, mul_44, view_117, addmm_34, view_119, mul_49, mul_52, view_123, view_139, mul_55, view_141, view_143, bmm_14, amax_7, sum_8, view_155, mul_58, view_157, addmm_44, view_159, mul_63, view_161, view_177, mul_66, view_179, bmm_18, amax_9, sum_10, view_193, mul_69, view_195, addmm_54, view_197, mul_74, view_199, view_215, mul_77, view_217, bmm_22, amax_11, sum_12, view_231, mul_80, view_233, addmm_64, view_235, mul_85, view_237, view_253, mul_88, view_255, bmm_26, amax_13, sum_14, view_269, mul_91, view_271, addmm_74, view_273, mul_96, view_275, view_291, mul_99, view_293, bmm_30, amax_15, sum_16, view_307, mul_102, view_309, addmm_84, view_311, mul_107, view_313, view_329, mul_110, view_331, bmm_34, amax_17, sum_18, view_345, mul_113, view_347, addmm_94, view_349, mul_118, view_351, permute_189, div_18, permute_191, permute_195, div_19, permute_199, permute_204, permute_205, permute_206, permute_207, permute_211, permute_216, permute_220, div_20, permute_224, permute_229, permute_230, alias_19, permute_231, permute_232, permute_236, permute_241, permute_245, div_21, permute_249, permute_253, div_22, permute_257, permute_262, permute_263, permute_264, permute_265, permute_269, permute_274, permute_278, div_23, permute_282, permute_287, permute_288, alias_21, permute_289, permute_290, permute_294, permute_299, permute_303, div_24, permute_307, permute_311, div_25, permute_315, permute_320, permute_321, permute_322, permute_323, permute_327, permute_332, permute_336, div_26, permute_340, permute_345, permute_346, alias_23, permute_347, permute_348, permute_352, permute_357, permute_361, div_27, permute_365, permute_369, div_28, permute_373, permute_378, permute_379, permute_380, permute_381, permute_385, permute_390, permute_394, div_29, permute_398, permute_403, permute_404, alias_25, permute_405, permute_406, permute_410, permute_415, permute_419, div_30, permute_423, permute_427, div_31, permute_431, permute_436, permute_437, permute_438, permute_439, permute_443, permute_448, permute_452, div_32, permute_456, permute_461, permute_462, alias_27, permute_463, permute_464, permute_468, permute_473, permute_477, div_33, permute_481, permute_485, div_34, permute_489, permute_494, permute_495, permute_496, permute_497, permute_501, permute_506, permute_510, div_35, permute_514, permute_519, permute_520, alias_29, permute_521, permute_522, permute_526, permute_531, permute_535, div_36, div_37, permute_539, permute_543, div_38, permute_547, permute_552, permute_553, permute_554, permute_555, permute_559, permute_564, permute_568, div_39, permute_572, permute_576, div_40, permute_580, permute_585, permute_586, permute_587, permute_588, permute_592, permute_597, permute_601, div_41, permute_605, permute_609, div_42, permute_613, permute_618, permute_619, permute_620, permute_621, permute_625, permute_630, permute_634, div_43, permute_638, permute_642, div_44, permute_646, permute_651, permute_652, permute_653, permute_654, permute_658, permute_663, permute_667, div_45, permute_671, permute_675, div_46, permute_679, permute_684, permute_685, permute_686, permute_687, permute_691, permute_696, permute_700, div_47, permute_704, permute_708, div_48, permute_712, permute_717, permute_718, permute_719, permute_720, permute_724, permute_729, permute_733, div_49, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bart', benchmark_compiled_module)
