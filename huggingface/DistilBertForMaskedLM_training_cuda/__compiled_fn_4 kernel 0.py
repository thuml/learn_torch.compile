
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


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzvhodqb7y2i76kitxxumwib2vrzzstvjdlg3und3jb4pz2t6mh.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_6
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/oj/cojib5cjtva2qww365umgv3zoxwow4fqvapujgxd7ejhr7zbxrys.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_6
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssvjxto64knro2v4hwyn6ryjg2pmaies6l7ridcecucd4jh6qhd.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 7631
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x1 = (xindex // 4)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7631*x0)
        tmp1 = tl.full([1, 1], 30522, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r2 + (7631*x0) + (30522*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.full([1, 1], -100, tl.int64)
        tmp6 = tmp4 != tmp5
        tmp11 = tmp8 / tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp3 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqeqd4obdtct7s6hhvg7aywslt5qz5v34w4kedmovsuq5iqdlhs.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxojzzanortxwjlfsxqwmmi2c5rur7ge6trunvbsfab74rws27w.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 30522)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp13 = tl.load(in_ptr5 + (x2), xmask)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 - tmp16
    tmp18 = tmp0 + tmp17
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuq2xqmgk7rim7af2w57h46z74p4ktmsklbtsuzgsw6p63dn3sbt.py
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (x0 + (30522*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3fnllh3uoueq3ry6iy2kcrsebzhgyknugyu5yh7lm7ntjsvfuj.py
# Source Nodes: [prediction_logits_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# prediction_logits_1 => add_45, erf_6, mul_45
triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = 0.7071067811865476
    tmp22 = tmp20 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp20
    tmp29 = -0.5
    tmp30 = tmp28 * tmp29
    tmp31 = tl.exp(tmp30)
    tmp32 = 0.3989422804014327
    tmp33 = tmp31 * tmp32
    tmp34 = tmp20 * tmp33
    tmp35 = tmp27 + tmp34
    tmp36 = tmp19 * tmp35
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoakzbihom7ms7vytcyadn46ey6unrnplk3pclrzp72xvh4bf7q.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ud/cudpdecwzfhknhomacx5rfbjntbofbwaya77jxb2hmjuceuuj6xs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszwxosfxzfzn75jpb7nmkrqnqoxck3vn7tf7uprvee73rpc7nk3.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp24, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwrkwomcsgxi5a3sxrq5ekvgpwbklpjj7fi2n4mgnq3uxkvu2ng.py
# Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
# x_21 => add_41, erf_5, mul_40
triton_poi_fused_gelu_gelu_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrzz2w3bwa5vkq2siqrgssekhq62y6phlycn7yn4kqcol2vskcu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvoswxnpuc634dpiu4ztes3hlocq6ilih7fue7dmnk7rcrqhyuq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2wb6o7faw23kitael2fxgjhukdzjygknhzyhiw522kqkw6fo4j.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/of/cof2wr2g4e4hhhhtazjcfsv4mpo6r6hwakqnl3dsq26niv35f5du.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlsvkfgo4ohc3zxgo5jgp2ucltmigngobk5qrszzosg42ivpo4n.py
# Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp6 * tmp11
    tmp14 = tmp7 - tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp12, tmp15, tmp14)
    tl.store(out_ptr1 + (r1 + (128*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6kuni52bw7phk7maq4a2kw22kn4mov76peel5muwnsvswdmqpb.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (8192*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymgmvakt5js44takqdkerwqk3obz6l6wbazm2u7daiuhy2higuq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5uburg7gxvl5a2l4clsb4b2ltwrdh3bkyt2bsixqqawj2uppsx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/th/cthwypzp4c6zlmwyofghkkt5qay6vzebtjzuc6zywxza3px6z6j2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfccvw5dxq23pywxnpwbzs5obkbaaxmudwvvobrkzdogvj7o6zc6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaqcbewe7mv4k65cazcbxxhqwv6vjsp76uak6psxfz3sswiprhk.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_7
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 128
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp13 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 768.0
    tmp26 = tmp13 * tmp25
    tmp27 = tmp26 - tmp17
    tmp28 = tmp18 * tmp23
    tmp29 = tmp27 - tmp28
    tmp30 = tmp24 * tmp29
    tmp32 = tl.full([1], -1, tl.int64)
    tmp33 = tmp31 == tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp37 = tl.full([1], 0, tl.int64)
    tmp38 = tmp36 == tmp37
    tmp39 = tl.where(tmp38, tmp34, tmp30)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c373wqik4ocfgmyp6rxsgpmcurd4cvgqmydqheoiplfkz4xuk4xm.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgo55u7zkr2kmi5je4v2paoclbxck4nr62blqo5tzq7gsqwz3fqr.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_103, primals_108, primals_109, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, view_138, addmm_36, mul_47, view_140, sub_21, convert_element_type, permute_68, div_14, permute_72, div_15, permute_76, permute_80, div_16, permute_84, permute_89, permute_90, alias_8, permute_91, permute_92, permute_95, permute_100, permute_105, div_18, permute_109, permute_113, div_19, permute_117, permute_122, permute_123, alias_9, permute_124, permute_125, permute_128, permute_133, permute_138, div_21, permute_142, permute_146, div_22, permute_150, permute_155, permute_156, alias_10, permute_157, permute_158, permute_161, permute_166, permute_171, div_24, permute_175, permute_179, div_25, permute_183, permute_188, permute_189, alias_11, permute_190, permute_191, permute_194, permute_199, permute_204, div_27, permute_208, permute_212, div_28, permute_216, permute_221, permute_222, alias_12, permute_223, permute_224, permute_227, permute_232, permute_237, div_30, permute_241, permute_245, div_31, permute_249, permute_254, permute_255, alias_13, permute_256, permute_257, permute_260, permute_265, permute_270, div_33, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_108, (1, 128), (128, 1))
    assert_size_stride(primals_109, (1, 128), (128, 1))
    assert_size_stride(slice_2, (1, 128), (512, 1))
    assert_size_stride(mul, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_3, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (128, 768), (768, 1))
    assert_size_stride(view_12, (1, 1, 1, 128), (128, 128, 128, 1))
    assert_size_stride(getitem_5, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_17, (128, 768), (768, 1))
    assert_size_stride(mul_2, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_19, (128, 768), (768, 1))
    assert_size_stride(addmm_4, (128, 3072), (3072, 1))
    assert_size_stride(view_21, (128, 3072), (3072, 1))
    assert_size_stride(getitem_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_7, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_23, (128, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_40, (128, 768), (768, 1))
    assert_size_stride(mul_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_42, (128, 768), (768, 1))
    assert_size_stride(addmm_10, (128, 3072), (3072, 1))
    assert_size_stride(view_44, (128, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_14, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_46, (128, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_63, (128, 768), (768, 1))
    assert_size_stride(mul_16, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_65, (128, 768), (768, 1))
    assert_size_stride(addmm_16, (128, 3072), (3072, 1))
    assert_size_stride(view_67, (128, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_21, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_69, (128, 768), (768, 1))
    assert_size_stride(getitem_29, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_86, (128, 768), (768, 1))
    assert_size_stride(mul_23, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (128, 768), (768, 1))
    assert_size_stride(addmm_22, (128, 3072), (3072, 1))
    assert_size_stride(view_90, (128, 3072), (3072, 1))
    assert_size_stride(getitem_33, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_28, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_92, (128, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_109, (128, 768), (768, 1))
    assert_size_stride(mul_30, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_111, (128, 768), (768, 1))
    assert_size_stride(addmm_28, (128, 3072), (3072, 1))
    assert_size_stride(view_113, (128, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_35, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_115, (128, 768), (768, 1))
    assert_size_stride(getitem_45, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_132, (128, 768), (768, 1))
    assert_size_stride(mul_37, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_134, (128, 768), (768, 1))
    assert_size_stride(addmm_34, (128, 3072), (3072, 1))
    assert_size_stride(view_136, (128, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_42, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_138, (128, 768), (768, 1))
    assert_size_stride(addmm_36, (128, 768), (768, 1))
    assert_size_stride(mul_47, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_140, (128, 768), (768, 1))
    assert_size_stride(sub_21, (128, 30522), (30522, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_68, (30522, 768), (768, 1))
    assert_size_stride(div_14, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_72, (768, 768), (768, 1))
    assert_size_stride(div_15, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_76, (768, 3072), (3072, 1))
    assert_size_stride(permute_80, (3072, 768), (768, 1))
    assert_size_stride(div_16, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_84, (768, 768), (768, 1))
    assert_size_stride(permute_89, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_90, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_8, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_91, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_92, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_95, (768, 768), (768, 1))
    assert_size_stride(permute_100, (768, 768), (768, 1))
    assert_size_stride(permute_105, (768, 768), (768, 1))
    assert_size_stride(div_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_109, (768, 3072), (3072, 1))
    assert_size_stride(permute_113, (3072, 768), (768, 1))
    assert_size_stride(div_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_117, (768, 768), (768, 1))
    assert_size_stride(permute_122, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_123, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_9, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_124, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_125, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_128, (768, 768), (768, 1))
    assert_size_stride(permute_133, (768, 768), (768, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_155, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_156, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_10, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_157, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_158, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_188, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_189, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_11, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_190, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_191, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_221, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_222, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_12, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_223, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_224, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_232, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_254, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_255, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_256, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_257, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_265, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 30522), (3906816, 30522, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((128, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 3906816, grid=grid(3906816), stream=stream0)
        buf1 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_109, buf1, 128, grid=grid(128), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf4 = empty_strided((128, 1, 4), (4, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_109, tangents_1, convert_element_type, buf4, 512, 7631, grid=grid(512), stream=stream0)
        buf5 = empty_strided((128, 1), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_per_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_3.run(buf4, buf5, 128, 4, grid=grid(128), stream=stream0)
        del buf4
        buf3 = empty((128, 30522), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf3, (1, 128, 30522), (3906816, 30522, 1), 0); del buf3  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_4.run(buf6, tangents_2, buf0, primals_109, tangents_1, convert_element_type, sub_21, buf5, 3906816, grid=grid(3906816), stream=stream0)
        del buf0
        del buf5
        del convert_element_type
        del primals_109
        del sub_21
        del tangents_1
        del tangents_2
        buf7 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 30522), (30522, 1), 0), permute_68, out=buf7)
        del permute_68
        buf8 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (30522, 128), (1, 30522), 0), view_140, out=buf8)
        del view_140
        buf9 = empty((1, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf6, buf9, 30522, 128, grid=grid(30522), stream=stream0)
        del buf6
        buf14 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [prediction_logits_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_6.run(buf7, primals_103, mul_47, div_14, addmm_36, buf14, 128, 768, grid=grid(128), stream=stream0)
        del addmm_36
        del div_14
        del primals_103
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        buf13 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_7.run(buf7, mul_47, buf12, buf13, 768, 128, grid=grid(768), stream=stream0)
        del mul_47
        buf15 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (128, 768), (768, 1), 0), permute_72, out=buf15)
        del permute_72
        buf16 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (768, 128), (1, 768), 0), view_138, out=buf16)
        del view_138
        buf17 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf14, buf17, 768, 128, grid=grid(768), stream=stream0)
        buf20 = buf14; del buf14  # reuse
        buf23 = empty((1, 128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_9.run(buf15, primals_99, mul_42, div_15, getitem_49, buf20, buf23, 128, 768, grid=grid(128), stream=stream0)
        del div_15
        del getitem_49
        del primals_99
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_7.run(buf15, mul_42, buf21, buf22, 768, 128, grid=grid(768), stream=stream0)
        del mul_42
        buf24 = empty((128, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (128, 768), (768, 1), 0), permute_76, out=buf24)
        del permute_76
        buf25 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (768, 128), (1, 768), 0), view_136, out=buf25)
        del view_136
        buf26 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf23, buf26, 768, 128, grid=grid(768), stream=stream0)
        buf27 = reinterpret_tensor(buf24, (1, 128, 3072), (393216, 3072, 1), 0); del buf24  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf27, addmm_34, 393216, grid=grid(393216), stream=stream0)
        del addmm_34
        buf28 = reinterpret_tensor(buf23, (128, 768), (768, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (128, 3072), (3072, 1), 0), permute_80, out=buf28)
        del permute_80
        buf29 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (3072, 128), (1, 3072), 0), view_134, out=buf29)
        del view_134
        buf30 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf27, buf30, 3072, 128, grid=grid(3072), stream=stream0)
        buf33 = reinterpret_tensor(buf15, (1, 128, 768), (98304, 768, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf20, buf28, primals_93, mul_37, div_16, buf33, 128, 768, grid=grid(128), stream=stream0)
        del div_16
        del primals_93
        buf34 = empty((768, ), device='cuda', dtype=torch.float32)
        buf35 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf20, buf28, mul_37, buf34, buf35, 768, 128, grid=grid(768), stream=stream0)
        del mul_37
        buf36 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (128, 768), (768, 1), 0), permute_84, out=buf36)
        del permute_84
        buf37 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (768, 128), (1, 768), 0), view_132, out=buf37)
        del view_132
        buf39 = reinterpret_tensor(buf20, (12, 128, 64), (8192, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_89, reinterpret_tensor(buf36, (12, 128, 64), (64, 768, 1), 0), out=buf39)
        del permute_89
        buf45 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf39, buf45, 98304, grid=grid(98304), stream=stream0)
        buf46 = reinterpret_tensor(buf39, (128, 768), (768, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, permute_95, out=buf46)
        del permute_95
        buf40 = empty((12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (12, 128, 64), (64, 768, 1), 0), permute_90, out=buf40)
        del permute_90
        buf42 = empty((1, 12, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf40, getitem_45, alias_8, view_12, buf42, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_8
        del getitem_45
        buf43 = reinterpret_tensor(buf36, (12, 64, 128), (8192, 128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_91, reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0), out=buf43)
        del permute_91
        buf49 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (128, 768), (1, 128), 0), permute_100, out=buf49)
        del permute_100
        buf44 = empty((12, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0), permute_92, out=buf44)
        del permute_92
        buf52 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf44, buf52, 98304, grid=grid(98304), stream=stream0)
        buf53 = reinterpret_tensor(buf44, (128, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, permute_105, out=buf53)
        del permute_105
        buf38 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf60 = empty((768, ), device='cuda', dtype=torch.float32)
        buf61 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf33, buf46, buf49, buf53, mul_35, buf38, buf60, buf61, 768, 128, grid=grid(768), stream=stream0)
        buf47 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (768, 128), (1, 768), 0), view_115, out=buf47)
        buf48 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf45, buf48, 768, 128, grid=grid(768), stream=stream0)
        buf50 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (768, 128), (128, 1), 0), view_115, out=buf50)
        buf51 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf43, buf51, 768, 128, grid=grid(768), stream=stream0)
        buf54 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (768, 128), (1, 768), 0), view_115, out=buf54)
        del view_115
        buf55 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf52, buf55, 768, 128, grid=grid(768), stream=stream0)
        buf56 = buf33; del buf33  # reuse
        buf59 = reinterpret_tensor(buf52, (1, 128, 768), (98304, 768, 1), 0); del buf52  # reuse
        buf62 = reinterpret_tensor(buf43, (1, 128, 768), (98304, 768, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf56, buf46, buf49, buf53, primals_83, mul_35, div_18, getitem_41, buf59, buf62, 128, 768, grid=grid(128), stream=stream0)
        del div_18
        del getitem_41
        del mul_35
        del primals_83
        buf63 = reinterpret_tensor(buf27, (128, 3072), (3072, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (128, 768), (768, 1), 0), permute_109, out=buf63)
        del permute_109
        buf64 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (768, 128), (1, 768), 0), view_113, out=buf64)
        del view_113
        buf65 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf62, buf65, 768, 128, grid=grid(768), stream=stream0)
        buf66 = reinterpret_tensor(buf63, (1, 128, 3072), (393216, 3072, 1), 0); del buf63  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf66, addmm_28, 393216, grid=grid(393216), stream=stream0)
        del addmm_28
        buf67 = reinterpret_tensor(buf62, (128, 768), (768, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (128, 3072), (3072, 1), 0), permute_113, out=buf67)
        del permute_113
        buf68 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (3072, 128), (1, 3072), 0), view_111, out=buf68)
        del view_111
        buf69 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf66, buf69, 3072, 128, grid=grid(3072), stream=stream0)
        buf72 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf59, buf67, primals_77, mul_30, div_19, buf72, 128, 768, grid=grid(128), stream=stream0)
        del div_19
        del primals_77
        buf73 = empty((768, ), device='cuda', dtype=torch.float32)
        buf74 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf59, buf67, mul_30, buf73, buf74, 768, 128, grid=grid(768), stream=stream0)
        del mul_30
        buf75 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (128, 768), (768, 1), 0), permute_117, out=buf75)
        del permute_117
        buf76 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (768, 128), (1, 768), 0), view_109, out=buf76)
        del view_109
        buf78 = reinterpret_tensor(buf59, (12, 128, 64), (8192, 64, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_122, reinterpret_tensor(buf75, (12, 128, 64), (64, 768, 1), 0), out=buf78)
        del permute_122
        buf84 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf78, buf84, 98304, grid=grid(98304), stream=stream0)
        buf85 = reinterpret_tensor(buf78, (128, 768), (768, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf84, permute_128, out=buf85)
        del permute_128
        buf79 = reinterpret_tensor(buf42, (12, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf75, (12, 128, 64), (64, 768, 1), 0), permute_123, out=buf79)
        del permute_123
        buf81 = reinterpret_tensor(buf40, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf40  # reuse
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf79, getitem_37, alias_9, view_12, buf81, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_9
        del getitem_37
        buf82 = reinterpret_tensor(buf75, (12, 64, 128), (8192, 128, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_124, reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0), out=buf82)
        del permute_124
        buf88 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (128, 768), (1, 128), 0), permute_133, out=buf88)
        del permute_133
        buf83 = reinterpret_tensor(buf46, (12, 128, 64), (8192, 64, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0), permute_125, out=buf83)
        del permute_125
        buf91 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf83, buf91, 98304, grid=grid(98304), stream=stream0)
        buf92 = reinterpret_tensor(buf83, (128, 768), (768, 1), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf91, permute_138, out=buf92)
        del permute_138
        buf77 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf99 = empty((768, ), device='cuda', dtype=torch.float32)
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf72, buf85, buf88, buf92, mul_28, buf77, buf99, buf100, 768, 128, grid=grid(768), stream=stream0)
        buf86 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (768, 128), (1, 768), 0), view_92, out=buf86)
        buf87 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf84, buf87, 768, 128, grid=grid(768), stream=stream0)
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (768, 128), (128, 1), 0), view_92, out=buf89)
        buf90 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf82, buf90, 768, 128, grid=grid(768), stream=stream0)
        buf93 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (768, 128), (1, 768), 0), view_92, out=buf93)
        del view_92
        buf94 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf91, buf94, 768, 128, grid=grid(768), stream=stream0)
        buf95 = buf72; del buf72  # reuse
        buf98 = reinterpret_tensor(buf91, (1, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
        buf101 = reinterpret_tensor(buf82, (1, 128, 768), (98304, 768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf95, buf85, buf88, buf92, primals_67, mul_28, div_21, getitem_33, buf98, buf101, 128, 768, grid=grid(128), stream=stream0)
        del div_21
        del getitem_33
        del mul_28
        del primals_67
        buf102 = reinterpret_tensor(buf66, (128, 3072), (3072, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (128, 768), (768, 1), 0), permute_142, out=buf102)
        del permute_142
        buf103 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (768, 128), (1, 768), 0), view_90, out=buf103)
        del view_90
        buf104 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf101, buf104, 768, 128, grid=grid(768), stream=stream0)
        buf105 = reinterpret_tensor(buf102, (1, 128, 3072), (393216, 3072, 1), 0); del buf102  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf105, addmm_22, 393216, grid=grid(393216), stream=stream0)
        del addmm_22
        buf106 = reinterpret_tensor(buf101, (128, 768), (768, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (128, 3072), (3072, 1), 0), permute_146, out=buf106)
        del permute_146
        buf107 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (3072, 128), (1, 3072), 0), view_88, out=buf107)
        del view_88
        buf108 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf105, buf108, 3072, 128, grid=grid(3072), stream=stream0)
        buf111 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf98, buf106, primals_61, mul_23, div_22, buf111, 128, 768, grid=grid(128), stream=stream0)
        del div_22
        del primals_61
        buf112 = empty((768, ), device='cuda', dtype=torch.float32)
        buf113 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf98, buf106, mul_23, buf112, buf113, 768, 128, grid=grid(768), stream=stream0)
        del mul_23
        buf114 = reinterpret_tensor(buf98, (128, 768), (768, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (128, 768), (768, 1), 0), permute_150, out=buf114)
        del permute_150
        buf115 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (768, 128), (1, 768), 0), view_86, out=buf115)
        del view_86
        buf117 = reinterpret_tensor(buf106, (12, 128, 64), (8192, 64, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_155, reinterpret_tensor(buf114, (12, 128, 64), (64, 768, 1), 0), out=buf117)
        del permute_155
        buf123 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf117, buf123, 98304, grid=grid(98304), stream=stream0)
        buf124 = reinterpret_tensor(buf117, (128, 768), (768, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf123, permute_161, out=buf124)
        del permute_161
        buf118 = reinterpret_tensor(buf81, (12, 128, 128), (16384, 128, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (12, 128, 64), (64, 768, 1), 0), permute_156, out=buf118)
        del permute_156
        buf120 = reinterpret_tensor(buf79, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf79  # reuse
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf118, getitem_29, alias_10, view_12, buf120, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_10
        del getitem_29
        buf121 = reinterpret_tensor(buf114, (12, 64, 128), (8192, 128, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_157, reinterpret_tensor(buf120, (12, 128, 128), (16384, 128, 1), 0), out=buf121)
        del permute_157
        buf127 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (128, 768), (1, 128), 0), permute_166, out=buf127)
        del permute_166
        buf122 = reinterpret_tensor(buf85, (12, 128, 64), (8192, 64, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (12, 128, 128), (16384, 128, 1), 0), permute_158, out=buf122)
        del permute_158
        buf130 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf122, buf130, 98304, grid=grid(98304), stream=stream0)
        buf131 = reinterpret_tensor(buf122, (128, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf130, permute_171, out=buf131)
        del permute_171
        buf116 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf138 = empty((768, ), device='cuda', dtype=torch.float32)
        buf139 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf111, buf124, buf127, buf131, mul_21, buf116, buf138, buf139, 768, 128, grid=grid(768), stream=stream0)
        buf125 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (768, 128), (1, 768), 0), view_69, out=buf125)
        buf126 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf123, buf126, 768, 128, grid=grid(768), stream=stream0)
        buf128 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (768, 128), (128, 1), 0), view_69, out=buf128)
        buf129 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf121, buf129, 768, 128, grid=grid(768), stream=stream0)
        buf132 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (768, 128), (1, 768), 0), view_69, out=buf132)
        del view_69
        buf133 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf130, buf133, 768, 128, grid=grid(768), stream=stream0)
        buf134 = buf111; del buf111  # reuse
        buf137 = reinterpret_tensor(buf130, (1, 128, 768), (98304, 768, 1), 0); del buf130  # reuse
        buf140 = reinterpret_tensor(buf121, (1, 128, 768), (98304, 768, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf134, buf124, buf127, buf131, primals_51, mul_21, div_24, getitem_25, buf137, buf140, 128, 768, grid=grid(128), stream=stream0)
        del div_24
        del getitem_25
        del mul_21
        del primals_51
        buf141 = reinterpret_tensor(buf105, (128, 3072), (3072, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (128, 768), (768, 1), 0), permute_175, out=buf141)
        del permute_175
        buf142 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (768, 128), (1, 768), 0), view_67, out=buf142)
        del view_67
        buf143 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf140, buf143, 768, 128, grid=grid(768), stream=stream0)
        buf144 = reinterpret_tensor(buf141, (1, 128, 3072), (393216, 3072, 1), 0); del buf141  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf144, addmm_16, 393216, grid=grid(393216), stream=stream0)
        del addmm_16
        buf145 = reinterpret_tensor(buf140, (128, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (128, 3072), (3072, 1), 0), permute_179, out=buf145)
        del permute_179
        buf146 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (3072, 128), (1, 3072), 0), view_65, out=buf146)
        del view_65
        buf147 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf144, buf147, 3072, 128, grid=grid(3072), stream=stream0)
        buf150 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf137, buf145, primals_45, mul_16, div_25, buf150, 128, 768, grid=grid(128), stream=stream0)
        del div_25
        del primals_45
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        buf152 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf137, buf145, mul_16, buf151, buf152, 768, 128, grid=grid(768), stream=stream0)
        del mul_16
        buf153 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (128, 768), (768, 1), 0), permute_183, out=buf153)
        del permute_183
        buf154 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (768, 128), (1, 768), 0), view_63, out=buf154)
        del view_63
        buf156 = reinterpret_tensor(buf137, (12, 128, 64), (8192, 64, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_188, reinterpret_tensor(buf153, (12, 128, 64), (64, 768, 1), 0), out=buf156)
        del permute_188
        buf162 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf156, buf162, 98304, grid=grid(98304), stream=stream0)
        buf163 = reinterpret_tensor(buf156, (128, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, permute_194, out=buf163)
        del permute_194
        buf157 = reinterpret_tensor(buf120, (12, 128, 128), (16384, 128, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (12, 128, 64), (64, 768, 1), 0), permute_189, out=buf157)
        del permute_189
        buf159 = reinterpret_tensor(buf118, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf118  # reuse
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf157, getitem_21, alias_11, view_12, buf159, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_11
        del getitem_21
        buf160 = reinterpret_tensor(buf153, (12, 64, 128), (8192, 128, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_190, reinterpret_tensor(buf159, (12, 128, 128), (16384, 128, 1), 0), out=buf160)
        del permute_190
        buf166 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (128, 768), (1, 128), 0), permute_199, out=buf166)
        del permute_199
        buf161 = reinterpret_tensor(buf124, (12, 128, 64), (8192, 64, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf159, (12, 128, 128), (16384, 128, 1), 0), permute_191, out=buf161)
        del permute_191
        buf169 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf161, buf169, 98304, grid=grid(98304), stream=stream0)
        buf170 = reinterpret_tensor(buf161, (128, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf169, permute_204, out=buf170)
        del permute_204
        buf155 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf177 = empty((768, ), device='cuda', dtype=torch.float32)
        buf178 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf150, buf163, buf166, buf170, mul_14, buf155, buf177, buf178, 768, 128, grid=grid(768), stream=stream0)
        buf164 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (768, 128), (1, 768), 0), view_46, out=buf164)
        buf165 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf162, buf165, 768, 128, grid=grid(768), stream=stream0)
        buf167 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (768, 128), (128, 1), 0), view_46, out=buf167)
        buf168 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf160, buf168, 768, 128, grid=grid(768), stream=stream0)
        buf171 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (768, 128), (1, 768), 0), view_46, out=buf171)
        del view_46
        buf172 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf169, buf172, 768, 128, grid=grid(768), stream=stream0)
        buf173 = buf150; del buf150  # reuse
        buf176 = reinterpret_tensor(buf169, (1, 128, 768), (98304, 768, 1), 0); del buf169  # reuse
        buf179 = reinterpret_tensor(buf160, (1, 128, 768), (98304, 768, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf173, buf163, buf166, buf170, primals_35, mul_14, div_27, getitem_17, buf176, buf179, 128, 768, grid=grid(128), stream=stream0)
        del div_27
        del getitem_17
        del mul_14
        del primals_35
        buf180 = reinterpret_tensor(buf144, (128, 3072), (3072, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (128, 768), (768, 1), 0), permute_208, out=buf180)
        del permute_208
        buf181 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (768, 128), (1, 768), 0), view_44, out=buf181)
        del view_44
        buf182 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf179, buf182, 768, 128, grid=grid(768), stream=stream0)
        buf183 = reinterpret_tensor(buf180, (1, 128, 3072), (393216, 3072, 1), 0); del buf180  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf183, addmm_10, 393216, grid=grid(393216), stream=stream0)
        del addmm_10
        buf184 = reinterpret_tensor(buf179, (128, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (128, 3072), (3072, 1), 0), permute_212, out=buf184)
        del permute_212
        buf185 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (3072, 128), (1, 3072), 0), view_42, out=buf185)
        del view_42
        buf186 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf183, buf186, 3072, 128, grid=grid(3072), stream=stream0)
        buf189 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf176, buf184, primals_29, mul_9, div_28, buf189, 128, 768, grid=grid(128), stream=stream0)
        del div_28
        del primals_29
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf176, buf184, mul_9, buf190, buf191, 768, 128, grid=grid(768), stream=stream0)
        del mul_9
        buf192 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (128, 768), (768, 1), 0), permute_216, out=buf192)
        del permute_216
        buf193 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (768, 128), (1, 768), 0), view_40, out=buf193)
        del view_40
        buf195 = reinterpret_tensor(buf176, (12, 128, 64), (8192, 64, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_221, reinterpret_tensor(buf192, (12, 128, 64), (64, 768, 1), 0), out=buf195)
        del permute_221
        buf201 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf195, buf201, 98304, grid=grid(98304), stream=stream0)
        buf202 = reinterpret_tensor(buf195, (128, 768), (768, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf201, permute_227, out=buf202)
        del permute_227
        buf196 = reinterpret_tensor(buf159, (12, 128, 128), (16384, 128, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (12, 128, 64), (64, 768, 1), 0), permute_222, out=buf196)
        del permute_222
        buf198 = reinterpret_tensor(buf157, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf157  # reuse
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf196, getitem_13, alias_12, view_12, buf198, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_12
        del getitem_13
        buf199 = reinterpret_tensor(buf192, (12, 64, 128), (8192, 128, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_223, reinterpret_tensor(buf198, (12, 128, 128), (16384, 128, 1), 0), out=buf199)
        del permute_223
        buf205 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (128, 768), (1, 128), 0), permute_232, out=buf205)
        del permute_232
        buf200 = reinterpret_tensor(buf163, (12, 128, 64), (8192, 64, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (12, 128, 128), (16384, 128, 1), 0), permute_224, out=buf200)
        del permute_224
        buf208 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf200, buf208, 98304, grid=grid(98304), stream=stream0)
        buf209 = reinterpret_tensor(buf200, (128, 768), (768, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf208, permute_237, out=buf209)
        del permute_237
        buf194 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf216 = empty((768, ), device='cuda', dtype=torch.float32)
        buf217 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_17.run(buf189, buf202, buf205, buf209, mul_7, buf194, buf216, buf217, 768, 128, grid=grid(768), stream=stream0)
        buf203 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (768, 128), (1, 768), 0), view_23, out=buf203)
        buf204 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf201, buf204, 768, 128, grid=grid(768), stream=stream0)
        del buf201
        buf206 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (768, 128), (128, 1), 0), view_23, out=buf206)
        buf207 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf199, buf207, 768, 128, grid=grid(768), stream=stream0)
        buf210 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (768, 128), (1, 768), 0), view_23, out=buf210)
        del view_23
        buf211 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf208, buf211, 768, 128, grid=grid(768), stream=stream0)
        buf212 = buf189; del buf189  # reuse
        buf215 = reinterpret_tensor(buf208, (1, 128, 768), (98304, 768, 1), 0); del buf208  # reuse
        buf218 = reinterpret_tensor(buf199, (1, 128, 768), (98304, 768, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_19.run(buf212, buf202, buf205, buf209, primals_19, mul_7, div_30, getitem_9, buf215, buf218, 128, 768, grid=grid(128), stream=stream0)
        del div_30
        del getitem_9
        del mul_7
        del primals_19
        buf219 = reinterpret_tensor(buf183, (128, 3072), (3072, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf218, (128, 768), (768, 1), 0), permute_241, out=buf219)
        del permute_241
        buf220 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf218, (768, 128), (1, 768), 0), view_21, out=buf220)
        del view_21
        buf221 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf218, buf221, 768, 128, grid=grid(768), stream=stream0)
        buf222 = reinterpret_tensor(buf219, (1, 128, 3072), (393216, 3072, 1), 0); del buf219  # reuse
        # Source Nodes: [x_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_10.run(buf222, addmm_4, 393216, grid=grid(393216), stream=stream0)
        del addmm_4
        buf223 = reinterpret_tensor(buf218, (128, 768), (768, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (128, 3072), (3072, 1), 0), permute_245, out=buf223)
        del permute_245
        buf224 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (3072, 128), (1, 3072), 0), view_19, out=buf224)
        del view_19
        buf225 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf222, buf225, 3072, 128, grid=grid(3072), stream=stream0)
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf215, buf223, primals_13, mul_2, div_31, buf228, 128, 768, grid=grid(128), stream=stream0)
        del div_31
        del primals_13
        buf229 = empty((768, ), device='cuda', dtype=torch.float32)
        buf230 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf215, buf223, mul_2, buf229, buf230, 768, 128, grid=grid(768), stream=stream0)
        del mul_2
        buf231 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (128, 768), (768, 1), 0), permute_249, out=buf231)
        del permute_249
        buf232 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (768, 128), (1, 768), 0), view_17, out=buf232)
        del view_17
        buf233 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf228, buf233, 768, 128, grid=grid(768), stream=stream0)
        buf234 = reinterpret_tensor(buf215, (12, 128, 64), (8192, 64, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_254, reinterpret_tensor(buf231, (12, 128, 64), (64, 768, 1), 0), out=buf234)
        del permute_254
        buf235 = reinterpret_tensor(buf198, (12, 128, 128), (16384, 128, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf231, (12, 128, 64), (64, 768, 1), 0), permute_255, out=buf235)
        del permute_255
        buf237 = reinterpret_tensor(buf196, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf196  # reuse
        # Source Nodes: [loss], Original ATen: [aten._softmax_backward_data, aten.masked_fill, aten.native_dropout_backward, aten.nll_loss_forward]
        triton_per_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_15.run(buf235, getitem_5, alias_13, view_12, buf237, 1536, 128, grid=grid(1536), stream=stream0)
        del alias_13
        del buf235
        del getitem_5
        del view_12
        buf238 = reinterpret_tensor(buf231, (12, 64, 128), (8192, 128, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_256, reinterpret_tensor(buf237, (12, 128, 128), (16384, 128, 1), 0), out=buf238)
        del permute_256
        buf239 = reinterpret_tensor(buf209, (12, 128, 64), (8192, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (12, 128, 128), (16384, 128, 1), 0), permute_257, out=buf239)
        del buf237
        del permute_257
        buf240 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf234, buf240, 98304, grid=grid(98304), stream=stream0)
        buf241 = reinterpret_tensor(buf234, (128, 768), (768, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf240, permute_260, out=buf241)
        del permute_260
        buf242 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (768, 128), (1, 768), 0), view, out=buf242)
        buf243 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf240, buf243, 768, 128, grid=grid(768), stream=stream0)
        buf244 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (128, 768), (1, 128), 0), permute_265, out=buf244)
        del permute_265
        buf245 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (768, 128), (128, 1), 0), view, out=buf245)
        buf246 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_18.run(buf238, buf246, 768, 128, grid=grid(768), stream=stream0)
        buf247 = reinterpret_tensor(buf238, (128, 768), (768, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_16.run(buf239, buf247, 98304, grid=grid(98304), stream=stream0)
        buf248 = reinterpret_tensor(buf239, (128, 768), (768, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf247, permute_270, out=buf248)
        del permute_270
        buf249 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (768, 128), (1, 768), 0), view, out=buf249)
        del view
        buf250 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf247, buf250, 768, 128, grid=grid(768), stream=stream0)
        buf251 = buf228; del buf228  # reuse
        buf258 = reinterpret_tensor(buf247, (1, 128, 768), (98304, 768, 1), 0); del buf247  # reuse
        buf262 = reinterpret_tensor(buf202, (1, 128, 768), (98304, 768, 1), 0); del buf202  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_21.run(buf251, buf241, buf244, buf248, getitem_3, primals_3, mul, div_33, slice_2, primals_108, buf258, buf262, 128, 768, grid=grid(128), stream=stream0)
        del buf241
        del buf244
        del buf248
        del div_33
        del getitem_3
        del primals_3
        buf255 = empty((768, ), device='cuda', dtype=torch.float32)
        buf256 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_7.run(buf251, mul, buf255, buf256, 768, 128, grid=grid(768), stream=stream0)
        del buf251
        del mul
        buf257 = reinterpret_tensor(buf222, (512, 768), (768, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf257, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf257, [slice_2], buf258, True)
        del buf258
        del slice_2
        buf261 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_23.run(buf261, 23440896, grid=grid(23440896), stream=stream0)
        aten.index_put_(buf261, [primals_108], buf262, True)
        del buf262
        del primals_108
        return (buf261, buf257, buf255, buf256, reinterpret_tensor(buf249, (768, 768), (768, 1), 0), reinterpret_tensor(buf250, (768, ), (1, ), 0), reinterpret_tensor(buf245, (768, 768), (768, 1), 0), reinterpret_tensor(buf246, (768, ), (1, ), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf243, (768, ), (1, ), 0), reinterpret_tensor(buf232, (768, 768), (768, 1), 0), reinterpret_tensor(buf233, (768, ), (1, ), 0), buf229, buf230, reinterpret_tensor(buf224, (3072, 768), (768, 1), 0), reinterpret_tensor(buf225, (3072, ), (1, ), 0), reinterpret_tensor(buf220, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf221, (768, ), (1, ), 0), buf216, buf217, reinterpret_tensor(buf210, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (768, ), (1, ), 0), reinterpret_tensor(buf206, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf203, (768, 768), (768, 1), 0), reinterpret_tensor(buf204, (768, ), (1, ), 0), reinterpret_tensor(buf193, (768, 768), (768, 1), 0), reinterpret_tensor(buf194, (768, ), (1, ), 0), buf190, buf191, reinterpret_tensor(buf185, (3072, 768), (768, 1), 0), reinterpret_tensor(buf186, (3072, ), (1, ), 0), reinterpret_tensor(buf181, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf182, (768, ), (1, ), 0), buf177, buf178, reinterpret_tensor(buf171, (768, 768), (768, 1), 0), reinterpret_tensor(buf172, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, 768), (768, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), reinterpret_tensor(buf164, (768, 768), (768, 1), 0), reinterpret_tensor(buf165, (768, ), (1, ), 0), reinterpret_tensor(buf154, (768, 768), (768, 1), 0), reinterpret_tensor(buf155, (768, ), (1, ), 0), buf151, buf152, reinterpret_tensor(buf146, (3072, 768), (768, 1), 0), reinterpret_tensor(buf147, (3072, ), (1, ), 0), reinterpret_tensor(buf142, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf143, (768, ), (1, ), 0), buf138, buf139, reinterpret_tensor(buf132, (768, 768), (768, 1), 0), reinterpret_tensor(buf133, (768, ), (1, ), 0), reinterpret_tensor(buf128, (768, 768), (768, 1), 0), reinterpret_tensor(buf129, (768, ), (1, ), 0), reinterpret_tensor(buf125, (768, 768), (768, 1), 0), reinterpret_tensor(buf126, (768, ), (1, ), 0), reinterpret_tensor(buf115, (768, 768), (768, 1), 0), reinterpret_tensor(buf116, (768, ), (1, ), 0), buf112, buf113, reinterpret_tensor(buf107, (3072, 768), (768, 1), 0), reinterpret_tensor(buf108, (3072, ), (1, ), 0), reinterpret_tensor(buf103, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf104, (768, ), (1, ), 0), buf99, buf100, reinterpret_tensor(buf93, (768, 768), (768, 1), 0), reinterpret_tensor(buf94, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, 768), (768, 1), 0), reinterpret_tensor(buf90, (768, ), (1, ), 0), reinterpret_tensor(buf86, (768, 768), (768, 1), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf76, (768, 768), (768, 1), 0), reinterpret_tensor(buf77, (768, ), (1, ), 0), buf73, buf74, reinterpret_tensor(buf68, (3072, 768), (768, 1), 0), reinterpret_tensor(buf69, (3072, ), (1, ), 0), reinterpret_tensor(buf64, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf65, (768, ), (1, ), 0), buf60, buf61, reinterpret_tensor(buf54, (768, 768), (768, 1), 0), reinterpret_tensor(buf55, (768, ), (1, ), 0), reinterpret_tensor(buf50, (768, 768), (768, 1), 0), reinterpret_tensor(buf51, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, 768), (768, 1), 0), reinterpret_tensor(buf48, (768, ), (1, ), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), buf34, buf35, reinterpret_tensor(buf29, (3072, 768), (768, 1), 0), reinterpret_tensor(buf30, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), buf21, buf22, reinterpret_tensor(buf16, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf8, (30522, 768), (768, 1), 0), reinterpret_tensor(buf9, (30522, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_109 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    slice_2 = rand_strided((1, 128), (512, 1), device='cuda:0', dtype=torch.int64)
    mul = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((1, 1, 1, 128), (128, 128, 128, 1), device='cuda:0', dtype=torch.bool)
    getitem_5 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_17 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_7 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_40 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_14 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_63 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_21 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_86 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_28 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_109 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_35 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_45 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_132 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_138 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_47 = rand_strided((1, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_21 = rand_strided((128, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_89 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_8 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_92 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_113 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_9 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_11 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((12, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((12, 64, 128), (64, 1, 768), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((12, 128, 64), (64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 30522), (3906816, 30522, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_103, primals_108, primals_109, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, view_138, addmm_36, mul_47, view_140, sub_21, convert_element_type, permute_68, div_14, permute_72, div_15, permute_76, permute_80, div_16, permute_84, permute_89, permute_90, alias_8, permute_91, permute_92, permute_95, permute_100, permute_105, div_18, permute_109, permute_113, div_19, permute_117, permute_122, permute_123, alias_9, permute_124, permute_125, permute_128, permute_133, permute_138, div_21, permute_142, permute_146, div_22, permute_150, permute_155, permute_156, alias_10, permute_157, permute_158, permute_161, permute_166, permute_171, div_24, permute_175, permute_179, div_25, permute_183, permute_188, permute_189, alias_11, permute_190, permute_191, permute_194, permute_199, permute_204, div_27, permute_208, permute_212, div_28, permute_216, permute_221, permute_222, alias_12, permute_223, permute_224, permute_227, permute_232, permute_237, div_30, permute_241, permute_245, div_31, permute_249, permute_254, permute_255, alias_13, permute_256, permute_257, permute_260, permute_265, permute_270, div_33, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForMaskedLM', benchmark_compiled_module)
