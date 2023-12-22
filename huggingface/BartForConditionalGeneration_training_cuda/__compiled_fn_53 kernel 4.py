
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


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfnifpknglnwhrxrmk6gvpwzqydnmhffqwoq4lzjue3tu5yuhub.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp24, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/dy/cdymzoxnim6aod4vjqbvocwtpmifnn2p2t6qel36hkv3xatlng7o.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3or4wlh5ztetc6nnsthk5becvbecb3eewi5urbs6rznn5uzplfe.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnhgpjjfyhyulm4davyldxnpyvlejcqvsceetka7co4vzvfmmdvn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/es/cesa4m4a5fzvudind3pk2iriihadkqwrztcwlmf2wfg47l76y5a5.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_8 => add_7, erf, mul_7
triton_poi_fused_gelu_gelu_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5siyybbixcdolzeb4z6yznvnu236gavjzsnlvyjmismvvf4hdl.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (524288*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hjvab4f45wkabkvcxvz2i2lsj64yf73na5h5tw5qcyuhinksst.py
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
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctsmrkigviwjetxnxh3dx3ookaqcyrggz2x3d5mraswque7c25xc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csokrv2f32aeesm3gevrklmydqqermzqqp2unutta4a4k7tjrtl6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqgdzvbu74kbieumijfw2i3mep4jr7ivos7rycxrrxnwikmkyhft.py
# Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
# attn_weights_5 => div_1, exp_1, sub_2
triton_per_fused__softmax__softmax_backward_data_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
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
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/col4prcc6j7xaluv2xtl6xdwvsyv2pjzdk6vgrogprxjxkwwc6kj.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz5pyftzwxie4s4yjbizttfoqzxjjbiwl745uodt7y37r437jdx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7j54ir3bwzppnrzyit3v4y2rempu3oqtiai3uo36sk3prpestg.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmlp6x6wb4n4mxz7a5zlerqecrhpgvtaqjq67ed4lduipsnsoxo.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjsux4lgtmgpqqafgdhn6zkuydvvbh7sizrj6jwiraprhtgdjiv.py
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3hkgoxyx5ee2jqsmzl47ngobya2nsbxvj5pet3iq7engfjgwiwu.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_9, primals_19, primals_25, view, view_16, getitem_1, mul_1, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_5, mul_4, view_34, addmm_8, view_36, getitem_9, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1 = args
    args.clear()
    assert_size_stride(primals_9, (1024, ), (1, ))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(view, (1024, 1024), (1024, 1))
    assert_size_stride(view_16, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_1, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(mul_1, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(view_18, (1024, 1024), (1024, 1))
    assert_size_stride(view_20, (1024, 1024), (1024, 1))
    assert_size_stride(bmm_2, (16, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(amax_1, (16, 1024, 1), (1024, 1, 1))
    assert_size_stride(sum_2, (16, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_32, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_5, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(mul_4, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(view_34, (1024, 1024), (1024, 1))
    assert_size_stride(addmm_8, (1024, 4096), (4096, 1))
    assert_size_stride(view_36, (1024, 4096), (4096, 1))
    assert_size_stride(getitem_9, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(mul_9, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(div_2, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_20, (1024, 4096), (4096, 1))
    assert_size_stride(permute_24, (4096, 1024), (1024, 1))
    assert_size_stride(div_3, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_28, (1024, 1024), (1024, 1))
    assert_size_stride(permute_33, (16, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_34, (16, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_35, (16, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_36, (16, 1024, 64), (65536, 64, 1))
    assert_size_stride(permute_40, (1024, 1024), (1024, 1))
    assert_size_stride(permute_45, (1024, 1024), (1024, 1))
    assert_size_stride(permute_49, (1024, 1024), (1024, 1))
    assert_size_stride(div_4, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_53, (1024, 1024), (1024, 1))
    assert_size_stride(permute_58, (16, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_59, (16, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_3, (16, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(permute_60, (16, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_61, (16, 1024, 64), (65536, 64, 1))
    assert_size_stride(permute_65, (1024, 1024), (1024, 1))
    assert_size_stride(permute_70, (1024, 1024), (1024, 1))
    assert_size_stride(permute_74, (1024, 1024), (1024, 1))
    assert_size_stride(tangents_1, (1, 1024, 1024), (1048576, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, primals_25, mul_9, div_2, getitem_9, buf2, buf5, 1024, 1024, grid=grid(1024), stream=stream0)
        del div_2
        del getitem_9
        del primals_25
        buf3 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf4 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(tangents_1, mul_9, buf3, buf4, 1024, 1024, grid=grid(1024), stream=stream0)
        del mul_9
        del tangents_1
        buf6 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 1024), (1024, 1), 0), permute_20, out=buf6)
        del permute_20
        buf7 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 1024), (1, 1024), 0), view_36, out=buf7)
        del view_36
        buf8 = empty_strided((1, 1024, 8), (8192, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf5, buf8, 8192, 128, grid=grid(8192), stream=stream0)
        buf9 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf8, buf9, 1024, 8, grid=grid(1024), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (1, 1024, 4096), (4194304, 4096, 1), 0); del buf6  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf10, addmm_8, 4194304, grid=grid(4194304), stream=stream0)
        del addmm_8
        buf11 = reinterpret_tensor(buf5, (1024, 1024), (1024, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1024, 4096), (4096, 1), 0), permute_24, out=buf11)
        del permute_24
        buf12 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (4096, 1024), (1, 4096), 0), view_34, out=buf12)
        del view_34
        buf13 = empty_strided((1, 4096, 8), (32768, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf10, buf13, 32768, 128, grid=grid(32768), stream=stream0)
        del buf10
        buf14 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf13, buf14, 4096, 8, grid=grid(4096), stream=stream0)
        del buf13
        buf17 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf20 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf2, buf11, primals_19, mul_4, div_3, getitem_5, buf17, buf20, 1024, 1024, grid=grid(1024), stream=stream0)
        del div_3
        del getitem_5
        del primals_19
        buf18 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf19 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf2, buf11, mul_4, buf18, buf19, 1024, 1024, grid=grid(1024), stream=stream0)
        del mul_4
        buf21 = reinterpret_tensor(buf2, (1024, 1024), (1024, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 1024), (1024, 1), 0), permute_28, out=buf21)
        del permute_28
        buf22 = buf11; del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 1024), (1, 1024), 0), view_32, out=buf22)
        del view_32
        buf23 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf20, buf23, 8192, 128, grid=grid(8192), stream=stream0)
        buf24 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf23, buf24, 1024, 8, grid=grid(1024), stream=stream0)
        buf25 = reinterpret_tensor(buf20, (16, 1024, 64), (65536, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_33, reinterpret_tensor(buf21, (16, 1024, 64), (64, 1024, 1), 0), out=buf25)
        del permute_33
        buf26 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (16, 1024, 64), (64, 1024, 1), 0), permute_34, out=buf26)
        del permute_34
        buf28 = empty((16, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_9.run(buf26, bmm_2, amax_1, sum_2, buf28, 16384, 1024, grid=grid(16384), stream=stream0)
        del amax_1
        del bmm_2
        del sum_2
        buf29 = reinterpret_tensor(buf21, (16, 64, 1024), (65536, 1024, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_35, buf28, out=buf29)
        del permute_35
        buf30 = empty((16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf28, permute_36, out=buf30)
        del permute_36
        buf31 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf25, buf31, 1048576, grid=grid(1048576), stream=stream0)
        buf32 = reinterpret_tensor(buf25, (1024, 1024), (1024, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, permute_40, out=buf32)
        del permute_40
        buf33 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 1024), (1, 1024), 0), view_20, out=buf33)
        buf34 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf31, buf34, 8192, 128, grid=grid(8192), stream=stream0)
        buf35 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf34, buf35, 1024, 8, grid=grid(1024), stream=stream0)
        buf36 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1024, 1024), (1, 1024), 0), permute_45, out=buf36)
        del permute_45
        buf37 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1024, 1024), (1024, 1), 0), view_20, out=buf37)
        del view_20
        buf38 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf29, buf38, 1024, 1024, grid=grid(1024), stream=stream0)
        buf39 = reinterpret_tensor(buf32, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf39, buf36, 1048576, grid=grid(1048576), stream=stream0)
        buf40 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_13.run(buf30, buf40, 1048576, grid=grid(1048576), stream=stream0)
        buf41 = reinterpret_tensor(buf30, (1024, 1024), (1024, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, permute_49, out=buf41)
        del permute_49
        buf42 = reinterpret_tensor(buf29, (1024, 1024), (1024, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (1024, 1024), (1, 1024), 0), view_18, out=buf42)
        del view_18
        buf43 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf40, buf43, 8192, 128, grid=grid(8192), stream=stream0)
        buf44 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf43, buf44, 1024, 8, grid=grid(1024), stream=stream0)
        buf47 = reinterpret_tensor(buf40, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf40  # reuse
        buf50 = empty((1, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf17, buf41, primals_9, mul_1, div_4, getitem_1, buf47, buf50, 1024, 1024, grid=grid(1024), stream=stream0)
        del div_4
        del getitem_1
        del primals_9
        buf48 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf49 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf17, buf41, mul_1, buf48, buf49, 1024, 1024, grid=grid(1024), stream=stream0)
        del mul_1
        buf51 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1024, 1024), (1024, 1), 0), permute_53, out=buf51)
        del permute_53
        buf52 = reinterpret_tensor(buf17, (1024, 1024), (1024, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1024, 1024), (1, 1024), 0), view_16, out=buf52)
        del view_16
        buf53 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf50, buf53, 8192, 128, grid=grid(8192), stream=stream0)
        buf54 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf53, buf54, 1024, 8, grid=grid(1024), stream=stream0)
        buf55 = reinterpret_tensor(buf50, (16, 1024, 64), (65536, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_58, reinterpret_tensor(buf51, (16, 1024, 64), (64, 1024, 1), 0), out=buf55)
        del permute_58
        buf56 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (16, 1024, 64), (64, 1024, 1), 0), permute_59, out=buf56)
        del permute_59
        buf58 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf56, alias_3, buf58, 16384, 1024, grid=grid(16384), stream=stream0)
        del alias_3
        del buf56
        buf59 = reinterpret_tensor(buf51, (16, 64, 1024), (65536, 1024, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_60, reinterpret_tensor(buf58, (16, 1024, 1024), (1048576, 1024, 1), 0), out=buf59)
        del permute_60
        buf60 = empty((16, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (16, 1024, 1024), (1048576, 1024, 1), 0), permute_61, out=buf60)
        del buf58
        del permute_61
        buf61 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_10.run(buf55, buf61, 1048576, grid=grid(1048576), stream=stream0)
        buf62 = reinterpret_tensor(buf55, (1024, 1024), (1024, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf61, permute_65, out=buf62)
        del permute_65
        buf63 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 1024), (1, 1024), 0), view, out=buf63)
        buf64 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf61, buf64, 8192, 128, grid=grid(8192), stream=stream0)
        buf65 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf64, buf65, 1024, 8, grid=grid(1024), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (1024, 1024), (1, 1024), 0), permute_70, out=buf66)
        del permute_70
        buf67 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (1024, 1024), (1024, 1), 0), view, out=buf67)
        buf68 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf59, buf68, 1024, 1024, grid=grid(1024), stream=stream0)
        buf69 = reinterpret_tensor(buf59, (1024, 1024), (1024, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_13.run(buf60, buf69, 1048576, grid=grid(1048576), stream=stream0)
        buf70 = reinterpret_tensor(buf60, (1024, 1024), (1024, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf69, permute_74, out=buf70)
        del permute_74
        buf71 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 1024), (1, 1024), 0), view, out=buf71)
        del view
        buf72 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf69, buf72, 8192, 128, grid=grid(8192), stream=stream0)
        del buf69
        buf73 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf72, buf73, 1024, 8, grid=grid(1024), stream=stream0)
        del buf72
        buf74 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf74, buf62, buf66, buf70, 1048576, grid=grid(1048576), stream=stream0)
        return (reinterpret_tensor(buf71, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf73, (1024, ), (1, ), 0), reinterpret_tensor(buf67, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf68, (1024, ), (1, ), 0), reinterpret_tensor(buf63, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf65, (1024, ), (1, ), 0), reinterpret_tensor(buf52, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf54, (1024, ), (1, ), 0), buf48, buf49, reinterpret_tensor(buf42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf44, (1024, ), (1, ), 0), reinterpret_tensor(buf37, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf38, (1024, ), (1, ), 0), reinterpret_tensor(buf33, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf35, (1024, ), (1, ), 0), reinterpret_tensor(buf22, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf24, (1024, ), (1, ), 0), buf18, buf19, reinterpret_tensor(buf12, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf14, (4096, ), (1, ), 0), reinterpret_tensor(buf7, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf9, (1024, ), (1, ), 0), buf3, buf4, buf74, None, buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((16, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((16, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((16, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_4 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_20 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((16, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((16, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((16, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((16, 1024, 64), (65536, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_58 = rand_strided((16, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((16, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_3 = rand_strided((16, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((16, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((16, 1024, 64), (65536, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_19, primals_25, view, view_16, getitem_1, mul_1, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_5, mul_4, view_34, addmm_8, view_36, getitem_9, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
