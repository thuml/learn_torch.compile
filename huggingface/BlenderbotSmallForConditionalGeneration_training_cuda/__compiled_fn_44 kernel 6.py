
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


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ykhydrszebavrtvxpi4lilqrgv5be5xyu77waljutw3wjffio6.py
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
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp24, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/sh/cshgvryp4lsxiihbmaqern6o3onb5wtcmoo2unzrnk4ob253c3hu.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ez/cezlklgysztzhjxackcutujganevjfv3lms2ybgpbqxqxfzytezw.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuieffvfpzjkrmrza3gu3lkqrgtu4fhjkuy4b6qljszjoeazgnhr.py
# Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_8 => add_7, erf, mul_7
triton_poi_fused_gelu_gelu_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
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


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pwzvcdrkp7wxiqsqmnmbqqgirnggng76m4sv4mmnfjemw5fewo.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkrxjaa3iayq23bbgom4xnc6hoe3cr3mrj5h2ri6ozmrunpbtqd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cnuxbtbsdyfqtchn45cltnehz7frzb7ekq3tv5clmu75man676.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdl5govsv7esvhxac3xw7tuunt5yjkckfrvjesvkpr7a7ifxa6l.py
# Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
# attn_weights_5 => div_1, exp_1, sub_2
triton_per_fused__softmax__softmax_backward_data_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/czn6ro76g56u2tiq7kaf3udern5yo3ux44chdtljkdlasc6jf7rh.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (4096*(x0 // 32)) + (x0 % 32)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvxbphuhbvrtjaeeyhptmdve7s3dypxxf6obwriasvabmrkhnuz.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cam7l4e4vvgocw2qnrwouxx3spq5vbiflltynh477lpcrby2jxlz.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cic7xyqjt44l5svjjazimkx4jfnm2rsii3ucmr4aep37e5dlpx6s.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (4096*(x0 // 32)) + (x0 % 32)), None)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uqxp43jj6tua5qhcyp3e4aaxnf33hsysskfatdcecdxybbu73t.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobo3sz4rzriw2sllhbtnexet3deu32acayicdu3txd5qtugmmwn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(view, (128, 512), (512, 1))
    assert_size_stride(view_16, (128, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(mul_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_18, (128, 512), (512, 1))
    assert_size_stride(view_20, (128, 512), (512, 1))
    assert_size_stride(bmm_2, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(amax_1, (16, 128, 1), (128, 1, 1))
    assert_size_stride(sum_2, (16, 128, 1), (128, 1, 1))
    assert_size_stride(view_32, (128, 512), (512, 1))
    assert_size_stride(getitem_5, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(mul_4, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_34, (128, 512), (512, 1))
    assert_size_stride(addmm_8, (128, 2048), (2048, 1))
    assert_size_stride(view_36, (128, 2048), (2048, 1))
    assert_size_stride(getitem_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(mul_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(div_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_20, (512, 2048), (2048, 1))
    assert_size_stride(permute_24, (2048, 512), (512, 1))
    assert_size_stride(div_3, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_28, (512, 512), (512, 1))
    assert_size_stride(permute_33, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_34, (16, 32, 128), (4096, 1, 32))
    assert_size_stride(permute_35, (16, 32, 128), (4096, 1, 32))
    assert_size_stride(permute_36, (16, 128, 32), (4096, 32, 1))
    assert_size_stride(permute_40, (512, 512), (512, 1))
    assert_size_stride(permute_45, (512, 512), (512, 1))
    assert_size_stride(permute_49, (512, 512), (512, 1))
    assert_size_stride(div_4, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_53, (512, 512), (512, 1))
    assert_size_stride(permute_58, (16, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_59, (16, 32, 128), (4096, 1, 32))
    assert_size_stride(alias_3, (16, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_60, (16, 32, 128), (4096, 1, 32))
    assert_size_stride(permute_61, (16, 128, 32), (4096, 32, 1))
    assert_size_stride(permute_65, (512, 512), (512, 1))
    assert_size_stride(permute_70, (512, 512), (512, 1))
    assert_size_stride(permute_74, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (1, 128, 512), (65536, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, primals_25, mul_9, div_2, getitem_9, buf2, buf5, 128, 512, grid=grid(128), stream=stream0)
        del div_2
        del getitem_9
        del primals_25
        buf3 = empty((512, ), device='cuda', dtype=torch.float32)
        buf4 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(tangents_1, mul_9, buf3, buf4, 512, 128, grid=grid(512), stream=stream0)
        del mul_9
        del tangents_1
        buf6 = empty((128, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (128, 512), (512, 1), 0), permute_20, out=buf6)
        del permute_20
        buf7 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 128), (1, 512), 0), view_36, out=buf7)
        del view_36
        buf8 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf5, buf8, 512, 128, grid=grid(512), stream=stream0)
        buf9 = reinterpret_tensor(buf6, (1, 128, 2048), (262144, 2048, 1), 0); del buf6  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_3.run(buf9, addmm_8, 262144, grid=grid(262144), stream=stream0)
        del addmm_8
        buf10 = reinterpret_tensor(buf5, (128, 512), (512, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (128, 2048), (2048, 1), 0), permute_24, out=buf10)
        del permute_24
        buf11 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (2048, 128), (1, 2048), 0), view_34, out=buf11)
        del view_34
        buf12 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf9, buf12, 2048, 128, grid=grid(2048), stream=stream0)
        buf15 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_5.run(buf2, buf10, primals_19, mul_4, div_3, getitem_5, buf15, buf18, 128, 512, grid=grid(128), stream=stream0)
        del div_3
        del getitem_5
        del primals_19
        buf16 = empty((512, ), device='cuda', dtype=torch.float32)
        buf17 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_6.run(buf2, buf10, mul_4, buf16, buf17, 512, 128, grid=grid(512), stream=stream0)
        del mul_4
        buf19 = reinterpret_tensor(buf2, (128, 512), (512, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (128, 512), (512, 1), 0), permute_28, out=buf19)
        del permute_28
        buf20 = reinterpret_tensor(buf9, (512, 512), (512, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (512, 128), (1, 512), 0), view_32, out=buf20)
        del view_32
        buf21 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf18, buf21, 512, 128, grid=grid(512), stream=stream0)
        buf22 = reinterpret_tensor(buf18, (16, 128, 32), (4096, 32, 1), 0); del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_33, reinterpret_tensor(buf19, (16, 128, 32), (32, 512, 1), 0), out=buf22)
        del permute_33
        buf23 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf19, (16, 128, 32), (32, 512, 1), 0), permute_34, out=buf23)
        del permute_34
        buf25 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_weights_5], Original ATen: [aten._softmax, aten._softmax_backward_data]
        triton_per_fused__softmax__softmax_backward_data_7.run(buf23, bmm_2, amax_1, sum_2, buf25, 2048, 128, grid=grid(2048), stream=stream0)
        del amax_1
        del bmm_2
        del sum_2
        buf26 = reinterpret_tensor(buf19, (16, 32, 128), (4096, 128, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_35, buf25, out=buf26)
        del permute_35
        buf27 = reinterpret_tensor(buf10, (16, 128, 32), (4096, 32, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf25, permute_36, out=buf27)
        del permute_36
        buf28 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf22, buf28, 65536, grid=grid(65536), stream=stream0)
        buf29 = reinterpret_tensor(buf22, (128, 512), (512, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, permute_40, out=buf29)
        del permute_40
        buf30 = reinterpret_tensor(buf25, (512, 512), (512, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 128), (1, 512), 0), view_20, out=buf30)
        buf31 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf28, buf31, 512, 128, grid=grid(512), stream=stream0)
        buf32 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (128, 512), (1, 128), 0), permute_45, out=buf32)
        del permute_45
        buf33 = reinterpret_tensor(buf23, (512, 512), (512, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (512, 128), (128, 1), 0), view_20, out=buf33)
        del view_20
        buf34 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf26, buf34, 512, 128, grid=grid(512), stream=stream0)
        buf35 = reinterpret_tensor(buf29, (1, 128, 512), (65536, 512, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf35, buf32, 65536, grid=grid(65536), stream=stream0)
        buf36 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_11.run(buf27, buf36, 65536, grid=grid(65536), stream=stream0)
        buf37 = reinterpret_tensor(buf27, (128, 512), (512, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf36, permute_49, out=buf37)
        del permute_49
        buf38 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (512, 128), (1, 512), 0), view_18, out=buf38)
        del view_18
        buf39 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf36, buf39, 512, 128, grid=grid(512), stream=stream0)
        buf42 = reinterpret_tensor(buf36, (1, 128, 512), (65536, 512, 1), 0); del buf36  # reuse
        buf45 = reinterpret_tensor(buf26, (1, 128, 512), (65536, 512, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_5.run(buf15, buf37, primals_9, mul_1, div_4, getitem_1, buf42, buf45, 128, 512, grid=grid(128), stream=stream0)
        del div_4
        del getitem_1
        del primals_9
        buf43 = empty((512, ), device='cuda', dtype=torch.float32)
        buf44 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_6.run(buf15, buf37, mul_1, buf43, buf44, 512, 128, grid=grid(512), stream=stream0)
        del mul_1
        buf46 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (128, 512), (512, 1), 0), permute_53, out=buf46)
        del permute_53
        buf47 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 128), (1, 512), 0), view_16, out=buf47)
        del view_16
        buf48 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf45, buf48, 512, 128, grid=grid(512), stream=stream0)
        buf49 = reinterpret_tensor(buf45, (16, 128, 32), (4096, 32, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_58, reinterpret_tensor(buf46, (16, 128, 32), (32, 512, 1), 0), out=buf49)
        del permute_58
        buf50 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (16, 128, 32), (32, 512, 1), 0), permute_59, out=buf50)
        del permute_59
        buf52 = empty((16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_12.run(buf50, alias_3, buf52, 2048, 128, grid=grid(2048), stream=stream0)
        del alias_3
        buf53 = reinterpret_tensor(buf46, (16, 32, 128), (4096, 128, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_60, reinterpret_tensor(buf52, (16, 128, 128), (16384, 128, 1), 0), out=buf53)
        del permute_60
        buf54 = reinterpret_tensor(buf15, (16, 128, 32), (4096, 32, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (16, 128, 128), (16384, 128, 1), 0), permute_61, out=buf54)
        del permute_61
        buf55 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf49, buf55, 65536, grid=grid(65536), stream=stream0)
        buf56 = reinterpret_tensor(buf49, (128, 512), (512, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, permute_65, out=buf56)
        del permute_65
        buf57 = reinterpret_tensor(buf52, (512, 512), (512, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (512, 128), (1, 512), 0), view, out=buf57)
        buf58 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf55, buf58, 512, 128, grid=grid(512), stream=stream0)
        buf59 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (128, 512), (1, 128), 0), permute_70, out=buf59)
        del permute_70
        buf60 = reinterpret_tensor(buf50, (512, 512), (512, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 128), (128, 1), 0), view, out=buf60)
        buf61 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf53, buf61, 512, 128, grid=grid(512), stream=stream0)
        buf62 = reinterpret_tensor(buf53, (128, 512), (512, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_11.run(buf54, buf62, 65536, grid=grid(65536), stream=stream0)
        buf63 = reinterpret_tensor(buf54, (128, 512), (512, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf62, permute_74, out=buf63)
        del permute_74
        buf64 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (512, 128), (1, 512), 0), view, out=buf64)
        del view
        buf65 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf62, buf65, 512, 128, grid=grid(512), stream=stream0)
        del buf62
        buf66 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf66, buf56, buf59, buf63, 65536, grid=grid(65536), stream=stream0)
        return (reinterpret_tensor(buf64, (512, 512), (512, 1), 0), reinterpret_tensor(buf65, (512, ), (1, ), 0), reinterpret_tensor(buf60, (512, 512), (512, 1), 0), reinterpret_tensor(buf61, (512, ), (1, ), 0), reinterpret_tensor(buf57, (512, 512), (512, 1), 0), reinterpret_tensor(buf58, (512, ), (1, ), 0), reinterpret_tensor(buf47, (512, 512), (512, 1), 0), reinterpret_tensor(buf48, (512, ), (1, ), 0), buf43, buf44, reinterpret_tensor(buf38, (512, 512), (512, 1), 0), reinterpret_tensor(buf39, (512, ), (1, ), 0), reinterpret_tensor(buf33, (512, 512), (512, 1), 0), reinterpret_tensor(buf34, (512, ), (1, ), 0), reinterpret_tensor(buf30, (512, 512), (512, 1), 0), reinterpret_tensor(buf31, (512, ), (1, ), 0), reinterpret_tensor(buf20, (512, 512), (512, 1), 0), reinterpret_tensor(buf21, (512, ), (1, ), 0), buf16, buf17, reinterpret_tensor(buf11, (2048, 512), (512, 1), 0), reinterpret_tensor(buf12, (2048, ), (1, ), 0), reinterpret_tensor(buf7, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf8, (512, ), (1, ), 0), buf3, buf4, buf66, None, buf35, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bmm_2 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((16, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    mul_4 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_20 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((16, 32, 128), (4096, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((16, 32, 128), (4096, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((16, 128, 32), (4096, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_58 = rand_strided((16, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((16, 32, 128), (4096, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_3 = rand_strided((16, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((16, 32, 128), (4096, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((16, 128, 32), (4096, 32, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_19, primals_25, view, view_16, getitem_1, mul_1, view_18, view_20, bmm_2, amax_1, sum_2, view_32, getitem_5, mul_4, view_34, addmm_8, view_36, getitem_9, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_33, permute_34, permute_35, permute_36, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
