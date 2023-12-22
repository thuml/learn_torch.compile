
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


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcnp3jsddkjevddf75vegkihuegdlazzollvvtw3o7o2t7wxwz6.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16386560
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjivva35nrjxummuzug4vgfwcnmnn5qo5bkjhdvca6ghm43la35h.py
# Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_1
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/so/csooqjdtxpqf7ztkinez4io4vwfgg3n3m4veok4eu7p2g2cuj7vz.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32005
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr4 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr0 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1 + (32005*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 * tmp13
        tmp30 = tmp26 - tmp29
        tmp31 = tmp15 + tmp30
        tl.store(out_ptr1 + (r1 + (32005*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcxccylbwg7dpvg3cyohgothiwj3cegwtsp22pxnlem7de42na6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32005
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (32005*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkpnwbrxhtprzbeskjsvcewrjkjck4lhud3yar2z5yhxiyfz623.py
# Source Nodes: [x_37], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# x_37 => add_102, erf_12, mul_89
triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/uj/cuj5jipddg65hcmq25dutvuyerqmzocp5m7uy3uhaqqpiq5zmj6v.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5e32ob646uag7xx6bcm77p3cbdhfinmau2k2fzb6vsgphniet6.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnu4hetcmyv2laomty752vknskibjjygllxzp5ciznvl4mugsv5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cghs7wjo7pdu56ewpdwlexdkoygzpigcwmovu2b3hljsza7id5pb.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/mi/cmig47we7pe6qi4m6fodzi7o7v6bw622yrvfq45hd42yuxtarlpf.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_98, erf_11, mul_84
triton_poi_fused_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yqq7sfjgaaxdkh5o3kqam47b6ywf4s5xn4mipbwryklhixmpic.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6ughq2lkyzherih2mehrcz7rlcve3vxx3kazka7vrbbtqebhve.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/ul/culdgxck7pkwghuuz27p4eono53pxybauflzgunao7v5tfq2fggg.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
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
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgq42whvhsx5nt2p6roncq4lq6d7l3nswd42piwxjdybpr4zjgds.py
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgn7acrylinhzaoqitio4kjmr54gvl6lr2n3s2sseowmc3tcqkv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
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
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkwhjkyss74y2efo7tkvswroscaw4hnuyir5rgzx5qibget3sre.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkbkzwysvlff7wrndlriuivpidnm7sezac5o6yhdfdgfirix3fe.py
# Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
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
    tmp40 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
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
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tmp31 == tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp33, tmp34, tmp30)
    tmp37 = tl.full([1], -1, tl.int64)
    tmp38 = tmp36 == tmp37
    tmp39 = tl.where(tmp38, tmp34, tmp30)
    tmp41 = tmp40 == tmp32
    tmp42 = tl.where(tmp41, tmp34, tmp30)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp42, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafcfdsa56ygbztwb2gvfr7jtt7tibt3l5o5v5fvrblepaptnyj5.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 394752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgb7zpu6vaajm6uuelt7gqlkcwwy65bgc4ofdq3lgpjsemsntwpf.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7cverqacyttyylmj7mk2j6tqmrzq7xx4twbyh6keoa4objxpop.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24579840
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_205, primals_206, expand, add_1, mul_2, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_4, view_18, addmm_4, view_20, getitem_11, mul_9, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_11, view_40, addmm_10, view_42, getitem_21, mul_16, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_18, view_62, addmm_16, view_64, getitem_31, mul_23, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_25, view_84, addmm_22, view_86, getitem_41, mul_30, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_32, view_106, addmm_28, view_108, getitem_51, mul_37, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_39, view_128, addmm_34, view_130, getitem_61, mul_44, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_46, view_150, addmm_40, view_152, getitem_71, mul_51, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_53, view_172, addmm_46, view_174, getitem_81, mul_58, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_60, view_194, addmm_52, view_196, getitem_91, mul_65, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_67, view_216, addmm_58, view_218, getitem_101, mul_72, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_74, view_238, addmm_64, view_240, getitem_111, mul_79, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_81, view_260, addmm_70, view_262, getitem_121, mul_86, view_264, addmm_72, mul_91, view_266, sub_40, convert_element_type_3, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2 = args
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
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_205, (1, 512), (512, 1))
    assert_size_stride(primals_206, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (514, 1))
    assert_size_stride(add_1, (1, 512), (512, 1))
    assert_size_stride(mul_2, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(clone_default_33, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_34, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_35, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_204, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_205, (), ())
    assert_size_stride(getitem_206, (), ())
    assert_size_stride(alias_default_23, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(clone_default_30, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_31, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_32, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_197, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_198, (), ())
    assert_size_stride(getitem_199, (), ())
    assert_size_stride(alias_default_21, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(clone_default_27, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_28, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_29, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_190, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_191, (), ())
    assert_size_stride(getitem_192, (), ())
    assert_size_stride(alias_default_19, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_18, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(clone_default_24, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_25, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_26, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_183, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_184, (), ())
    assert_size_stride(getitem_185, (), ())
    assert_size_stride(alias_default_17, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_30, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(clone_default_21, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_22, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_23, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_176, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_177, (), ())
    assert_size_stride(getitem_178, (), ())
    assert_size_stride(alias_default_15, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(clone_default_18, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_19, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_20, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_169, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_170, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(alias_default_13, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_39, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(clone_default_15, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_16, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_17, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_162, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_163, (), ())
    assert_size_stride(getitem_164, (), ())
    assert_size_stride(alias_default_11, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_46, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(clone_default_12, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_13, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_14, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_155, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(getitem_157, (), ())
    assert_size_stride(alias_default_9, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_53, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_58, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(clone_default_9, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_10, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_11, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_148, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_149, (), ())
    assert_size_stride(getitem_150, (), ())
    assert_size_stride(alias_default_7, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_60, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(clone_default_6, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_7, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_8, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_141, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_142, (), ())
    assert_size_stride(getitem_143, (), ())
    assert_size_stride(alias_default_5, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_72, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(clone_default_3, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_4, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_5, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_134, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(alias_default_3, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_74, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_79, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(clone_default, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_1, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_2, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_127, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(alias_default_1, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_86, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_264, (512, 768), (768, 1))
    assert_size_stride(addmm_72, (512, 768), (768, 1))
    assert_size_stride(mul_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_266, (512, 768), (768, 1))
    assert_size_stride(sub_40, (512, 32005), (32005, 1))
    assert_size_stride(convert_element_type_3, (), ())
    assert_size_stride(permute_134, (32005, 768), (768, 1))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_167, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_200, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 3072), (3072, 1))
    assert_size_stride(permute_344, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (768, 768), (768, 1))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_365, (768, 768), (768, 1))
    assert_size_stride(permute_369, (768, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 3072), (3072, 1))
    assert_size_stride(permute_377, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (768, 768), (768, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_402, (768, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (768, 3072), (3072, 1))
    assert_size_stride(permute_410, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (768, 768), (768, 1))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_435, (768, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (768, 3072), (3072, 1))
    assert_size_stride(permute_443, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (768, 768), (768, 1))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_464, (768, 768), (768, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (768, 3072), (3072, 1))
    assert_size_stride(permute_476, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (768, 768), (768, 1))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_497, (768, 768), (768, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (768, 3072), (3072, 1))
    assert_size_stride(permute_509, (3072, 768), (768, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (768, 768), (768, 1))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_530, (768, 768), (768, 1))
    assert_size_stride(permute_534, (768, 768), (768, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 32005), (16386560, 32005, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((512, 32005), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 16386560, grid=grid(16386560), stream=stream0)
        buf1 = empty_strided((512, 1), (1, 512), device='cuda', dtype=torch.int64)
        # Source Nodes: [loss], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_206, buf1, 512, grid=grid(512), stream=stream0)
        aten.scatter_(buf0,1,buf1,-1.0)
        del buf1
        buf5 = empty((1, 512, 32005), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_206, tangents_1, convert_element_type_3, tangents_2, sub_40, buf5, 512, 32005, grid=grid(512), stream=stream0)
        del buf0
        del convert_element_type_3
        del primals_206
        del sub_40
        del tangents_1
        del tangents_2
        buf6 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 32005), (32005, 1), 0), permute_134, out=buf6)
        del permute_134
        buf7 = empty((32005, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (32005, 512), (1, 32005), 0), view_266, out=buf7)
        del view_266
        buf8 = empty((1, 32005), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf5, buf8, 32005, 512, grid=grid(32005), stream=stream0)
        del buf5
        buf13 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf6, primals_200, mul_91, div_26, addmm_72, buf13, 512, 768, grid=grid(512), stream=stream0)
        del addmm_72
        del div_26
        del primals_200
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        buf12 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf6, mul_91, buf11, buf12, 768, 512, grid=grid(768), stream=stream0)
        del mul_91
        buf14 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 768), (768, 1), 0), permute_138, out=buf14)
        del permute_138
        buf15 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (768, 512), (1, 768), 0), view_264, out=buf15)
        del view_264
        buf16 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf13, buf16, 3072, 128, grid=grid(3072), stream=stream0)
        buf17 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf16, buf17, 768, 4, grid=grid(768), stream=stream0)
        buf20 = buf13; del buf13  # reuse
        buf23 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf14, primals_196, mul_86, div_27, getitem_121, buf20, buf23, 512, 768, grid=grid(512), stream=stream0)
        del div_27
        del getitem_121
        del primals_196
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf14, mul_86, buf21, buf22, 768, 512, grid=grid(768), stream=stream0)
        del mul_86
        buf24 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), (768, 1), 0), permute_142, out=buf24)
        del permute_142
        buf25 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (768, 512), (1, 768), 0), view_262, out=buf25)
        del view_262
        buf26 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf23, buf26, 3072, 128, grid=grid(3072), stream=stream0)
        buf27 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf26, buf27, 768, 4, grid=grid(768), stream=stream0)
        buf28 = reinterpret_tensor(buf24, (1, 512, 3072), (1572864, 3072, 1), 0); del buf24  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf28, addmm_70, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_70
        buf29 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 3072), (3072, 1), 0), permute_146, out=buf29)
        del permute_146
        buf30 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (3072, 512), (1, 3072), 0), view_260, out=buf30)
        del view_260
        buf31 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf28, buf31, 12288, 128, grid=grid(12288), stream=stream0)
        buf32 = reinterpret_tensor(buf26, (1, 3072), (3072, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf31, buf32, 3072, 4, grid=grid(3072), stream=stream0)
        buf35 = reinterpret_tensor(buf14, (1, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
        buf38 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf20, buf29, primals_190, mul_81, div_28, getitem_117, buf35, buf38, 512, 768, grid=grid(512), stream=stream0)
        del div_28
        del getitem_117
        del primals_190
        buf36 = empty((768, ), device='cuda', dtype=torch.float32)
        buf37 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf20, buf29, mul_81, buf36, buf37, 768, 512, grid=grid(768), stream=stream0)
        del buf20
        del mul_81
        buf39 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (512, 768), (768, 1), 0), permute_150, out=buf39)
        del permute_150
        buf40 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (768, 512), (1, 768), 0), view_258, out=buf40)
        del view_258
        buf41 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf38, buf41, 3072, 128, grid=grid(3072), stream=stream0)
        buf42 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf41, buf42, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf43 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf39, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_127
        del getitem_128
        del getitem_129
        buf44 = buf43[0]
        buf45 = buf43[1]
        buf46 = buf43[2]
        del buf43
        buf47 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 768), (768, 1), 0), permute_162, out=buf47)
        del permute_162
        buf48 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (768, 512), (1, 768), 0), view_242, out=buf48)
        buf49 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf46, buf49, 3072, 128, grid=grid(3072), stream=stream0)
        buf50 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf49, buf50, 768, 4, grid=grid(768), stream=stream0)
        buf51 = reinterpret_tensor(buf46, (512, 768), (768, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 768), (768, 1), 0), permute_167, out=buf51)
        del permute_167
        buf52 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (768, 512), (1, 768), 0), view_242, out=buf52)
        buf53 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf45, buf53, 3072, 128, grid=grid(3072), stream=stream0)
        buf54 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf53, buf54, 768, 4, grid=grid(768), stream=stream0)
        buf55 = reinterpret_tensor(buf45, (512, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 768), (768, 1), 0), permute_171, out=buf55)
        del permute_171
        buf56 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (768, 512), (1, 768), 0), view_242, out=buf56)
        del view_242
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf44, buf57, 3072, 128, grid=grid(3072), stream=stream0)
        buf58 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf57, buf58, 768, 4, grid=grid(768), stream=stream0)
        buf62 = reinterpret_tensor(buf44, (1, 512, 768), (393216, 768, 1), 0); del buf44  # reuse
        buf65 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf35, buf47, buf51, buf55, primals_180, mul_79, div_30, getitem_111, buf62, buf65, 512, 768, grid=grid(512), stream=stream0)
        del div_30
        del getitem_111
        del primals_180
        buf63 = empty((768, ), device='cuda', dtype=torch.float32)
        buf64 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf35, buf47, buf51, buf55, mul_79, buf63, buf64, 768, 512, grid=grid(768), stream=stream0)
        del buf35
        del buf47
        del mul_79
        buf66 = reinterpret_tensor(buf28, (512, 3072), (3072, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (512, 768), (768, 1), 0), permute_175, out=buf66)
        del permute_175
        buf67 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (768, 512), (1, 768), 0), view_240, out=buf67)
        del view_240
        buf68 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf65, buf68, 3072, 128, grid=grid(3072), stream=stream0)
        buf69 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf68, buf69, 768, 4, grid=grid(768), stream=stream0)
        buf70 = reinterpret_tensor(buf66, (1, 512, 3072), (1572864, 3072, 1), 0); del buf66  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf70, addmm_64, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_64
        buf71 = reinterpret_tensor(buf65, (512, 768), (768, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 3072), (3072, 1), 0), permute_179, out=buf71)
        del permute_179
        buf72 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (3072, 512), (1, 3072), 0), view_238, out=buf72)
        del view_238
        buf73 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf70, buf73, 12288, 128, grid=grid(12288), stream=stream0)
        buf74 = reinterpret_tensor(buf68, (1, 3072), (3072, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf73, buf74, 3072, 4, grid=grid(3072), stream=stream0)
        buf77 = reinterpret_tensor(buf55, (1, 512, 768), (393216, 768, 1), 0); del buf55  # reuse
        buf80 = reinterpret_tensor(buf51, (1, 512, 768), (393216, 768, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf62, buf71, primals_174, mul_74, div_31, getitem_107, buf77, buf80, 512, 768, grid=grid(512), stream=stream0)
        del div_31
        del getitem_107
        del primals_174
        buf78 = empty((768, ), device='cuda', dtype=torch.float32)
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf62, buf71, mul_74, buf78, buf79, 768, 512, grid=grid(768), stream=stream0)
        del buf62
        del mul_74
        buf81 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (512, 768), (768, 1), 0), permute_183, out=buf81)
        del permute_183
        buf82 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (768, 512), (1, 768), 0), view_236, out=buf82)
        del view_236
        buf83 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf80, buf83, 3072, 128, grid=grid(3072), stream=stream0)
        buf84 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf83, buf84, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf85 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf81, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_134
        del getitem_135
        del getitem_136
        buf86 = buf85[0]
        buf87 = buf85[1]
        buf88 = buf85[2]
        del buf85
        buf89 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (512, 768), (768, 1), 0), permute_195, out=buf89)
        del permute_195
        buf90 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (768, 512), (1, 768), 0), view_220, out=buf90)
        buf91 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf88, buf91, 3072, 128, grid=grid(3072), stream=stream0)
        buf92 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf91, buf92, 768, 4, grid=grid(768), stream=stream0)
        buf93 = reinterpret_tensor(buf88, (512, 768), (768, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), permute_200, out=buf93)
        del permute_200
        buf94 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (768, 512), (1, 768), 0), view_220, out=buf94)
        buf95 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf87, buf95, 3072, 128, grid=grid(3072), stream=stream0)
        buf96 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf95, buf96, 768, 4, grid=grid(768), stream=stream0)
        buf97 = reinterpret_tensor(buf87, (512, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (512, 768), (768, 1), 0), permute_204, out=buf97)
        del permute_204
        buf98 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (768, 512), (1, 768), 0), view_220, out=buf98)
        del view_220
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf86, buf99, 3072, 128, grid=grid(3072), stream=stream0)
        buf100 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf99, buf100, 768, 4, grid=grid(768), stream=stream0)
        buf104 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf107 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf77, buf89, buf93, buf97, primals_164, mul_72, div_33, getitem_101, buf104, buf107, 512, 768, grid=grid(512), stream=stream0)
        del div_33
        del getitem_101
        del primals_164
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        buf106 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf77, buf89, buf93, buf97, mul_72, buf105, buf106, 768, 512, grid=grid(768), stream=stream0)
        del buf77
        del buf89
        del mul_72
        buf108 = reinterpret_tensor(buf70, (512, 3072), (3072, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (512, 768), (768, 1), 0), permute_208, out=buf108)
        del permute_208
        buf109 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (768, 512), (1, 768), 0), view_218, out=buf109)
        del view_218
        buf110 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf107, buf110, 3072, 128, grid=grid(3072), stream=stream0)
        buf111 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf110, buf111, 768, 4, grid=grid(768), stream=stream0)
        buf112 = reinterpret_tensor(buf108, (1, 512, 3072), (1572864, 3072, 1), 0); del buf108  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf112, addmm_58, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_58
        buf113 = reinterpret_tensor(buf107, (512, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0), permute_212, out=buf113)
        del permute_212
        buf114 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (3072, 512), (1, 3072), 0), view_216, out=buf114)
        del view_216
        buf115 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf112, buf115, 12288, 128, grid=grid(12288), stream=stream0)
        buf116 = reinterpret_tensor(buf110, (1, 3072), (3072, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf115, buf116, 3072, 4, grid=grid(3072), stream=stream0)
        buf119 = reinterpret_tensor(buf97, (1, 512, 768), (393216, 768, 1), 0); del buf97  # reuse
        buf122 = reinterpret_tensor(buf93, (1, 512, 768), (393216, 768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf104, buf113, primals_158, mul_67, div_34, getitem_97, buf119, buf122, 512, 768, grid=grid(512), stream=stream0)
        del div_34
        del getitem_97
        del primals_158
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        buf121 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf104, buf113, mul_67, buf120, buf121, 768, 512, grid=grid(768), stream=stream0)
        del buf104
        del mul_67
        buf123 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (512, 768), (768, 1), 0), permute_216, out=buf123)
        del permute_216
        buf124 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (768, 512), (1, 768), 0), view_214, out=buf124)
        del view_214
        buf125 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf122, buf125, 3072, 128, grid=grid(3072), stream=stream0)
        buf126 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf125, buf126, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf127 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf123, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_141
        del getitem_142
        del getitem_143
        buf128 = buf127[0]
        buf129 = buf127[1]
        buf130 = buf127[2]
        del buf127
        buf131 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (512, 768), (768, 1), 0), permute_228, out=buf131)
        del permute_228
        buf132 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (768, 512), (1, 768), 0), view_198, out=buf132)
        buf133 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf130, buf133, 3072, 128, grid=grid(3072), stream=stream0)
        buf134 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf133, buf134, 768, 4, grid=grid(768), stream=stream0)
        buf135 = reinterpret_tensor(buf130, (512, 768), (768, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (512, 768), (768, 1), 0), permute_233, out=buf135)
        del permute_233
        buf136 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (768, 512), (1, 768), 0), view_198, out=buf136)
        buf137 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf129, buf137, 3072, 128, grid=grid(3072), stream=stream0)
        buf138 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf137, buf138, 768, 4, grid=grid(768), stream=stream0)
        buf139 = reinterpret_tensor(buf129, (512, 768), (768, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 768), (768, 1), 0), permute_237, out=buf139)
        del permute_237
        buf140 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (768, 512), (1, 768), 0), view_198, out=buf140)
        del view_198
        buf141 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf128, buf141, 3072, 128, grid=grid(3072), stream=stream0)
        buf142 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf141, buf142, 768, 4, grid=grid(768), stream=stream0)
        buf146 = reinterpret_tensor(buf128, (1, 512, 768), (393216, 768, 1), 0); del buf128  # reuse
        buf149 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf119, buf131, buf135, buf139, primals_148, mul_65, div_36, getitem_91, buf146, buf149, 512, 768, grid=grid(512), stream=stream0)
        del div_36
        del getitem_91
        del primals_148
        buf147 = empty((768, ), device='cuda', dtype=torch.float32)
        buf148 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf119, buf131, buf135, buf139, mul_65, buf147, buf148, 768, 512, grid=grid(768), stream=stream0)
        del buf119
        del buf131
        del mul_65
        buf150 = reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (512, 768), (768, 1), 0), permute_241, out=buf150)
        del permute_241
        buf151 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (768, 512), (1, 768), 0), view_196, out=buf151)
        del view_196
        buf152 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf149, buf152, 3072, 128, grid=grid(3072), stream=stream0)
        buf153 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf152, buf153, 768, 4, grid=grid(768), stream=stream0)
        buf154 = reinterpret_tensor(buf150, (1, 512, 3072), (1572864, 3072, 1), 0); del buf150  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf154, addmm_52, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_52
        buf155 = reinterpret_tensor(buf149, (512, 768), (768, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (512, 3072), (3072, 1), 0), permute_245, out=buf155)
        del permute_245
        buf156 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (3072, 512), (1, 3072), 0), view_194, out=buf156)
        del view_194
        buf157 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf154, buf157, 12288, 128, grid=grid(12288), stream=stream0)
        buf158 = reinterpret_tensor(buf152, (1, 3072), (3072, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf157, buf158, 3072, 4, grid=grid(3072), stream=stream0)
        buf161 = reinterpret_tensor(buf139, (1, 512, 768), (393216, 768, 1), 0); del buf139  # reuse
        buf164 = reinterpret_tensor(buf135, (1, 512, 768), (393216, 768, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf146, buf155, primals_142, mul_60, div_37, getitem_87, buf161, buf164, 512, 768, grid=grid(512), stream=stream0)
        del div_37
        del getitem_87
        del primals_142
        buf162 = empty((768, ), device='cuda', dtype=torch.float32)
        buf163 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf146, buf155, mul_60, buf162, buf163, 768, 512, grid=grid(768), stream=stream0)
        del buf146
        del mul_60
        buf165 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (512, 768), (768, 1), 0), permute_249, out=buf165)
        del permute_249
        buf166 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (768, 512), (1, 768), 0), view_192, out=buf166)
        del view_192
        buf167 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf164, buf167, 3072, 128, grid=grid(3072), stream=stream0)
        buf168 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf167, buf168, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf169 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf165, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_148
        del getitem_149
        del getitem_150
        buf170 = buf169[0]
        buf171 = buf169[1]
        buf172 = buf169[2]
        del buf169
        buf173 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (512, 768), (768, 1), 0), permute_261, out=buf173)
        del permute_261
        buf174 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (768, 512), (1, 768), 0), view_176, out=buf174)
        buf175 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf172, buf175, 3072, 128, grid=grid(3072), stream=stream0)
        buf176 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf175, buf176, 768, 4, grid=grid(768), stream=stream0)
        buf177 = reinterpret_tensor(buf172, (512, 768), (768, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 768), (768, 1), 0), permute_266, out=buf177)
        del permute_266
        buf178 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (768, 512), (1, 768), 0), view_176, out=buf178)
        buf179 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf171, buf179, 3072, 128, grid=grid(3072), stream=stream0)
        buf180 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf179, buf180, 768, 4, grid=grid(768), stream=stream0)
        buf181 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 768), (768, 1), 0), permute_270, out=buf181)
        del permute_270
        buf182 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (768, 512), (1, 768), 0), view_176, out=buf182)
        del view_176
        buf183 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf170, buf183, 3072, 128, grid=grid(3072), stream=stream0)
        buf184 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf183, buf184, 768, 4, grid=grid(768), stream=stream0)
        buf188 = reinterpret_tensor(buf170, (1, 512, 768), (393216, 768, 1), 0); del buf170  # reuse
        buf191 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf161, buf173, buf177, buf181, primals_132, mul_58, div_39, getitem_81, buf188, buf191, 512, 768, grid=grid(512), stream=stream0)
        del div_39
        del getitem_81
        del primals_132
        buf189 = empty((768, ), device='cuda', dtype=torch.float32)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf161, buf173, buf177, buf181, mul_58, buf189, buf190, 768, 512, grid=grid(768), stream=stream0)
        del buf161
        del buf173
        del mul_58
        buf192 = reinterpret_tensor(buf154, (512, 3072), (3072, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (512, 768), (768, 1), 0), permute_274, out=buf192)
        del permute_274
        buf193 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (768, 512), (1, 768), 0), view_174, out=buf193)
        del view_174
        buf194 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf191, buf194, 3072, 128, grid=grid(3072), stream=stream0)
        buf195 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf194, buf195, 768, 4, grid=grid(768), stream=stream0)
        buf196 = reinterpret_tensor(buf192, (1, 512, 3072), (1572864, 3072, 1), 0); del buf192  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf196, addmm_46, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_46
        buf197 = reinterpret_tensor(buf191, (512, 768), (768, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 3072), (3072, 1), 0), permute_278, out=buf197)
        del permute_278
        buf198 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (3072, 512), (1, 3072), 0), view_172, out=buf198)
        del view_172
        buf199 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf196, buf199, 12288, 128, grid=grid(12288), stream=stream0)
        buf200 = reinterpret_tensor(buf194, (1, 3072), (3072, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf199, buf200, 3072, 4, grid=grid(3072), stream=stream0)
        buf203 = reinterpret_tensor(buf181, (1, 512, 768), (393216, 768, 1), 0); del buf181  # reuse
        buf206 = reinterpret_tensor(buf177, (1, 512, 768), (393216, 768, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf188, buf197, primals_126, mul_53, div_40, getitem_77, buf203, buf206, 512, 768, grid=grid(512), stream=stream0)
        del div_40
        del getitem_77
        del primals_126
        buf204 = empty((768, ), device='cuda', dtype=torch.float32)
        buf205 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf188, buf197, mul_53, buf204, buf205, 768, 512, grid=grid(768), stream=stream0)
        del buf188
        del mul_53
        buf207 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (512, 768), (768, 1), 0), permute_282, out=buf207)
        del permute_282
        buf208 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (768, 512), (1, 768), 0), view_170, out=buf208)
        del view_170
        buf209 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf206, buf209, 3072, 128, grid=grid(3072), stream=stream0)
        buf210 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf209, buf210, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf211 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf207, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_155
        del getitem_156
        del getitem_157
        buf212 = buf211[0]
        buf213 = buf211[1]
        buf214 = buf211[2]
        del buf211
        buf215 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 768), (768, 1), 0), permute_294, out=buf215)
        del permute_294
        buf216 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (768, 512), (1, 768), 0), view_154, out=buf216)
        buf217 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf214, buf217, 3072, 128, grid=grid(3072), stream=stream0)
        buf218 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf217, buf218, 768, 4, grid=grid(768), stream=stream0)
        buf219 = reinterpret_tensor(buf214, (512, 768), (768, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (512, 768), (768, 1), 0), permute_299, out=buf219)
        del permute_299
        buf220 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (768, 512), (1, 768), 0), view_154, out=buf220)
        buf221 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf213, buf221, 3072, 128, grid=grid(3072), stream=stream0)
        buf222 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf221, buf222, 768, 4, grid=grid(768), stream=stream0)
        buf223 = reinterpret_tensor(buf213, (512, 768), (768, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 768), (768, 1), 0), permute_303, out=buf223)
        del permute_303
        buf224 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 512), (1, 768), 0), view_154, out=buf224)
        del view_154
        buf225 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf212, buf225, 3072, 128, grid=grid(3072), stream=stream0)
        buf226 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf225, buf226, 768, 4, grid=grid(768), stream=stream0)
        buf230 = reinterpret_tensor(buf212, (1, 512, 768), (393216, 768, 1), 0); del buf212  # reuse
        buf233 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf203, buf215, buf219, buf223, primals_116, mul_51, div_42, getitem_71, buf230, buf233, 512, 768, grid=grid(512), stream=stream0)
        del div_42
        del getitem_71
        del primals_116
        buf231 = empty((768, ), device='cuda', dtype=torch.float32)
        buf232 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf203, buf215, buf219, buf223, mul_51, buf231, buf232, 768, 512, grid=grid(768), stream=stream0)
        del buf203
        del buf215
        del mul_51
        buf234 = reinterpret_tensor(buf196, (512, 3072), (3072, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (512, 768), (768, 1), 0), permute_307, out=buf234)
        del permute_307
        buf235 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (768, 512), (1, 768), 0), view_152, out=buf235)
        del view_152
        buf236 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf233, buf236, 3072, 128, grid=grid(3072), stream=stream0)
        buf237 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf236, buf237, 768, 4, grid=grid(768), stream=stream0)
        buf238 = reinterpret_tensor(buf234, (1, 512, 3072), (1572864, 3072, 1), 0); del buf234  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf238, addmm_40, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_40
        buf239 = reinterpret_tensor(buf233, (512, 768), (768, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 3072), (3072, 1), 0), permute_311, out=buf239)
        del permute_311
        buf240 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (3072, 512), (1, 3072), 0), view_150, out=buf240)
        del view_150
        buf241 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf238, buf241, 12288, 128, grid=grid(12288), stream=stream0)
        buf242 = reinterpret_tensor(buf236, (1, 3072), (3072, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf241, buf242, 3072, 4, grid=grid(3072), stream=stream0)
        buf245 = reinterpret_tensor(buf223, (1, 512, 768), (393216, 768, 1), 0); del buf223  # reuse
        buf248 = reinterpret_tensor(buf219, (1, 512, 768), (393216, 768, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf230, buf239, primals_110, mul_46, div_43, getitem_67, buf245, buf248, 512, 768, grid=grid(512), stream=stream0)
        del div_43
        del getitem_67
        del primals_110
        buf246 = empty((768, ), device='cuda', dtype=torch.float32)
        buf247 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf230, buf239, mul_46, buf246, buf247, 768, 512, grid=grid(768), stream=stream0)
        del buf230
        del mul_46
        buf249 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 768), (768, 1), 0), permute_315, out=buf249)
        del permute_315
        buf250 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (768, 512), (1, 768), 0), view_148, out=buf250)
        del view_148
        buf251 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf248, buf251, 3072, 128, grid=grid(3072), stream=stream0)
        buf252 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf251, buf252, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf253 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf249, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_162
        del getitem_163
        del getitem_164
        buf254 = buf253[0]
        buf255 = buf253[1]
        buf256 = buf253[2]
        del buf253
        buf257 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (512, 768), (768, 1), 0), permute_327, out=buf257)
        del permute_327
        buf258 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (768, 512), (1, 768), 0), view_132, out=buf258)
        buf259 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf256, buf259, 3072, 128, grid=grid(3072), stream=stream0)
        buf260 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf259, buf260, 768, 4, grid=grid(768), stream=stream0)
        buf261 = reinterpret_tensor(buf256, (512, 768), (768, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 768), (768, 1), 0), permute_332, out=buf261)
        del permute_332
        buf262 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (768, 512), (1, 768), 0), view_132, out=buf262)
        buf263 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf255, buf263, 3072, 128, grid=grid(3072), stream=stream0)
        buf264 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf263, buf264, 768, 4, grid=grid(768), stream=stream0)
        buf265 = reinterpret_tensor(buf255, (512, 768), (768, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (512, 768), (768, 1), 0), permute_336, out=buf265)
        del permute_336
        buf266 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (768, 512), (1, 768), 0), view_132, out=buf266)
        del view_132
        buf267 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf254, buf267, 3072, 128, grid=grid(3072), stream=stream0)
        buf268 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf267, buf268, 768, 4, grid=grid(768), stream=stream0)
        buf272 = reinterpret_tensor(buf254, (1, 512, 768), (393216, 768, 1), 0); del buf254  # reuse
        buf275 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf245, buf257, buf261, buf265, primals_100, mul_44, div_45, getitem_61, buf272, buf275, 512, 768, grid=grid(512), stream=stream0)
        del div_45
        del getitem_61
        del primals_100
        buf273 = empty((768, ), device='cuda', dtype=torch.float32)
        buf274 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf245, buf257, buf261, buf265, mul_44, buf273, buf274, 768, 512, grid=grid(768), stream=stream0)
        del buf245
        del buf257
        del mul_44
        buf276 = reinterpret_tensor(buf238, (512, 3072), (3072, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (512, 768), (768, 1), 0), permute_340, out=buf276)
        del permute_340
        buf277 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (768, 512), (1, 768), 0), view_130, out=buf277)
        del view_130
        buf278 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf275, buf278, 3072, 128, grid=grid(3072), stream=stream0)
        buf279 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf278, buf279, 768, 4, grid=grid(768), stream=stream0)
        buf280 = reinterpret_tensor(buf276, (1, 512, 3072), (1572864, 3072, 1), 0); del buf276  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf280, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf281 = reinterpret_tensor(buf275, (512, 768), (768, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (512, 3072), (3072, 1), 0), permute_344, out=buf281)
        del permute_344
        buf282 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (3072, 512), (1, 3072), 0), view_128, out=buf282)
        del view_128
        buf283 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf280, buf283, 12288, 128, grid=grid(12288), stream=stream0)
        buf284 = reinterpret_tensor(buf278, (1, 3072), (3072, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf283, buf284, 3072, 4, grid=grid(3072), stream=stream0)
        buf287 = reinterpret_tensor(buf265, (1, 512, 768), (393216, 768, 1), 0); del buf265  # reuse
        buf290 = reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf272, buf281, primals_94, mul_39, div_46, getitem_57, buf287, buf290, 512, 768, grid=grid(512), stream=stream0)
        del div_46
        del getitem_57
        del primals_94
        buf288 = empty((768, ), device='cuda', dtype=torch.float32)
        buf289 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf272, buf281, mul_39, buf288, buf289, 768, 512, grid=grid(768), stream=stream0)
        del buf272
        del mul_39
        buf291 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (512, 768), (768, 1), 0), permute_348, out=buf291)
        del permute_348
        buf292 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (768, 512), (1, 768), 0), view_126, out=buf292)
        del view_126
        buf293 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf290, buf293, 3072, 128, grid=grid(3072), stream=stream0)
        buf294 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf293, buf294, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf295 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf291, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_169
        del getitem_170
        del getitem_171
        buf296 = buf295[0]
        buf297 = buf295[1]
        buf298 = buf295[2]
        del buf295
        buf299 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (512, 768), (768, 1), 0), permute_360, out=buf299)
        del permute_360
        buf300 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (768, 512), (1, 768), 0), view_110, out=buf300)
        buf301 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf298, buf301, 3072, 128, grid=grid(3072), stream=stream0)
        buf302 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf301, buf302, 768, 4, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf298, (512, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (512, 768), (768, 1), 0), permute_365, out=buf303)
        del permute_365
        buf304 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (768, 512), (1, 768), 0), view_110, out=buf304)
        buf305 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf297, buf305, 3072, 128, grid=grid(3072), stream=stream0)
        buf306 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf305, buf306, 768, 4, grid=grid(768), stream=stream0)
        buf307 = reinterpret_tensor(buf297, (512, 768), (768, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 768), (768, 1), 0), permute_369, out=buf307)
        del permute_369
        buf308 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (768, 512), (1, 768), 0), view_110, out=buf308)
        del view_110
        buf309 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf296, buf309, 3072, 128, grid=grid(3072), stream=stream0)
        buf310 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf309, buf310, 768, 4, grid=grid(768), stream=stream0)
        buf314 = reinterpret_tensor(buf296, (1, 512, 768), (393216, 768, 1), 0); del buf296  # reuse
        buf317 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf287, buf299, buf303, buf307, primals_84, mul_37, div_48, getitem_51, buf314, buf317, 512, 768, grid=grid(512), stream=stream0)
        del div_48
        del getitem_51
        del primals_84
        buf315 = empty((768, ), device='cuda', dtype=torch.float32)
        buf316 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf287, buf299, buf303, buf307, mul_37, buf315, buf316, 768, 512, grid=grid(768), stream=stream0)
        del buf287
        del buf299
        del mul_37
        buf318 = reinterpret_tensor(buf280, (512, 3072), (3072, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (512, 768), (768, 1), 0), permute_373, out=buf318)
        del permute_373
        buf319 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (768, 512), (1, 768), 0), view_108, out=buf319)
        del view_108
        buf320 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf317, buf320, 3072, 128, grid=grid(3072), stream=stream0)
        buf321 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf320, buf321, 768, 4, grid=grid(768), stream=stream0)
        buf322 = reinterpret_tensor(buf318, (1, 512, 3072), (1572864, 3072, 1), 0); del buf318  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf322, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf323 = reinterpret_tensor(buf317, (512, 768), (768, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (512, 3072), (3072, 1), 0), permute_377, out=buf323)
        del permute_377
        buf324 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (3072, 512), (1, 3072), 0), view_106, out=buf324)
        del view_106
        buf325 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf322, buf325, 12288, 128, grid=grid(12288), stream=stream0)
        buf326 = reinterpret_tensor(buf320, (1, 3072), (3072, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf325, buf326, 3072, 4, grid=grid(3072), stream=stream0)
        buf329 = reinterpret_tensor(buf307, (1, 512, 768), (393216, 768, 1), 0); del buf307  # reuse
        buf332 = reinterpret_tensor(buf303, (1, 512, 768), (393216, 768, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf314, buf323, primals_78, mul_32, div_49, getitem_47, buf329, buf332, 512, 768, grid=grid(512), stream=stream0)
        del div_49
        del getitem_47
        del primals_78
        buf330 = empty((768, ), device='cuda', dtype=torch.float32)
        buf331 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf314, buf323, mul_32, buf330, buf331, 768, 512, grid=grid(768), stream=stream0)
        del buf314
        del mul_32
        buf333 = buf323; del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (512, 768), (768, 1), 0), permute_381, out=buf333)
        del permute_381
        buf334 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 512), (1, 768), 0), view_104, out=buf334)
        del view_104
        buf335 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf332, buf335, 3072, 128, grid=grid(3072), stream=stream0)
        buf336 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf335, buf336, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf337 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf333, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_176
        del getitem_177
        del getitem_178
        buf338 = buf337[0]
        buf339 = buf337[1]
        buf340 = buf337[2]
        del buf337
        buf341 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 768), (768, 1), 0), permute_393, out=buf341)
        del permute_393
        buf342 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (768, 512), (1, 768), 0), view_88, out=buf342)
        buf343 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf340, buf343, 3072, 128, grid=grid(3072), stream=stream0)
        buf344 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf343, buf344, 768, 4, grid=grid(768), stream=stream0)
        buf345 = reinterpret_tensor(buf340, (512, 768), (768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 768), (768, 1), 0), permute_398, out=buf345)
        del permute_398
        buf346 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (768, 512), (1, 768), 0), view_88, out=buf346)
        buf347 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf339, buf347, 3072, 128, grid=grid(3072), stream=stream0)
        buf348 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf347, buf348, 768, 4, grid=grid(768), stream=stream0)
        buf349 = reinterpret_tensor(buf339, (512, 768), (768, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (512, 768), (768, 1), 0), permute_402, out=buf349)
        del permute_402
        buf350 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (768, 512), (1, 768), 0), view_88, out=buf350)
        del view_88
        buf351 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf338, buf351, 3072, 128, grid=grid(3072), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf351, buf352, 768, 4, grid=grid(768), stream=stream0)
        buf356 = reinterpret_tensor(buf338, (1, 512, 768), (393216, 768, 1), 0); del buf338  # reuse
        buf359 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf329, buf341, buf345, buf349, primals_68, mul_30, div_51, getitem_41, buf356, buf359, 512, 768, grid=grid(512), stream=stream0)
        del div_51
        del getitem_41
        del primals_68
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        buf358 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf329, buf341, buf345, buf349, mul_30, buf357, buf358, 768, 512, grid=grid(768), stream=stream0)
        del buf329
        del buf341
        del mul_30
        buf360 = reinterpret_tensor(buf322, (512, 3072), (3072, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (512, 768), (768, 1), 0), permute_406, out=buf360)
        del permute_406
        buf361 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (768, 512), (1, 768), 0), view_86, out=buf361)
        del view_86
        buf362 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf359, buf362, 3072, 128, grid=grid(3072), stream=stream0)
        buf363 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf362, buf363, 768, 4, grid=grid(768), stream=stream0)
        buf364 = reinterpret_tensor(buf360, (1, 512, 3072), (1572864, 3072, 1), 0); del buf360  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf364, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf365 = reinterpret_tensor(buf359, (512, 768), (768, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (512, 3072), (3072, 1), 0), permute_410, out=buf365)
        del permute_410
        buf366 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (3072, 512), (1, 3072), 0), view_84, out=buf366)
        del view_84
        buf367 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf364, buf367, 12288, 128, grid=grid(12288), stream=stream0)
        buf368 = reinterpret_tensor(buf362, (1, 3072), (3072, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf367, buf368, 3072, 4, grid=grid(3072), stream=stream0)
        buf371 = reinterpret_tensor(buf349, (1, 512, 768), (393216, 768, 1), 0); del buf349  # reuse
        buf374 = reinterpret_tensor(buf345, (1, 512, 768), (393216, 768, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf356, buf365, primals_62, mul_25, div_52, getitem_37, buf371, buf374, 512, 768, grid=grid(512), stream=stream0)
        del div_52
        del getitem_37
        del primals_62
        buf372 = empty((768, ), device='cuda', dtype=torch.float32)
        buf373 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf356, buf365, mul_25, buf372, buf373, 768, 512, grid=grid(768), stream=stream0)
        del buf356
        del mul_25
        buf375 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (512, 768), (768, 1), 0), permute_414, out=buf375)
        del permute_414
        buf376 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (768, 512), (1, 768), 0), view_82, out=buf376)
        del view_82
        buf377 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf374, buf377, 3072, 128, grid=grid(3072), stream=stream0)
        buf378 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf377, buf378, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf379 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf375, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_183
        del getitem_184
        del getitem_185
        buf380 = buf379[0]
        buf381 = buf379[1]
        buf382 = buf379[2]
        del buf379
        buf383 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (512, 768), (768, 1), 0), permute_426, out=buf383)
        del permute_426
        buf384 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (768, 512), (1, 768), 0), view_66, out=buf384)
        buf385 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf382, buf385, 3072, 128, grid=grid(3072), stream=stream0)
        buf386 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf385, buf386, 768, 4, grid=grid(768), stream=stream0)
        buf387 = reinterpret_tensor(buf382, (512, 768), (768, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (512, 768), (768, 1), 0), permute_431, out=buf387)
        del permute_431
        buf388 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (768, 512), (1, 768), 0), view_66, out=buf388)
        buf389 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf381, buf389, 3072, 128, grid=grid(3072), stream=stream0)
        buf390 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf389, buf390, 768, 4, grid=grid(768), stream=stream0)
        buf391 = reinterpret_tensor(buf381, (512, 768), (768, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 768), (768, 1), 0), permute_435, out=buf391)
        del permute_435
        buf392 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (768, 512), (1, 768), 0), view_66, out=buf392)
        del view_66
        buf393 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf380, buf393, 3072, 128, grid=grid(3072), stream=stream0)
        buf394 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf393, buf394, 768, 4, grid=grid(768), stream=stream0)
        buf398 = reinterpret_tensor(buf380, (1, 512, 768), (393216, 768, 1), 0); del buf380  # reuse
        buf401 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf371, buf383, buf387, buf391, primals_52, mul_23, div_54, getitem_31, buf398, buf401, 512, 768, grid=grid(512), stream=stream0)
        del div_54
        del getitem_31
        del primals_52
        buf399 = empty((768, ), device='cuda', dtype=torch.float32)
        buf400 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf371, buf383, buf387, buf391, mul_23, buf399, buf400, 768, 512, grid=grid(768), stream=stream0)
        del buf371
        del buf383
        del mul_23
        buf402 = reinterpret_tensor(buf364, (512, 3072), (3072, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (512, 768), (768, 1), 0), permute_439, out=buf402)
        del permute_439
        buf403 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (768, 512), (1, 768), 0), view_64, out=buf403)
        del view_64
        buf404 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf401, buf404, 3072, 128, grid=grid(3072), stream=stream0)
        buf405 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf404, buf405, 768, 4, grid=grid(768), stream=stream0)
        buf406 = reinterpret_tensor(buf402, (1, 512, 3072), (1572864, 3072, 1), 0); del buf402  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf406, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf407 = reinterpret_tensor(buf401, (512, 768), (768, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (512, 3072), (3072, 1), 0), permute_443, out=buf407)
        del permute_443
        buf408 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (3072, 512), (1, 3072), 0), view_62, out=buf408)
        del view_62
        buf409 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf406, buf409, 12288, 128, grid=grid(12288), stream=stream0)
        buf410 = reinterpret_tensor(buf404, (1, 3072), (3072, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf409, buf410, 3072, 4, grid=grid(3072), stream=stream0)
        buf413 = reinterpret_tensor(buf391, (1, 512, 768), (393216, 768, 1), 0); del buf391  # reuse
        buf416 = reinterpret_tensor(buf387, (1, 512, 768), (393216, 768, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf398, buf407, primals_46, mul_18, div_55, getitem_27, buf413, buf416, 512, 768, grid=grid(512), stream=stream0)
        del div_55
        del getitem_27
        del primals_46
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        buf415 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf398, buf407, mul_18, buf414, buf415, 768, 512, grid=grid(768), stream=stream0)
        del buf398
        del mul_18
        buf417 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (512, 768), (768, 1), 0), permute_447, out=buf417)
        del permute_447
        buf418 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (768, 512), (1, 768), 0), view_60, out=buf418)
        del view_60
        buf419 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf416, buf419, 3072, 128, grid=grid(3072), stream=stream0)
        buf420 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf419, buf420, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf421 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf417, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_190
        del getitem_191
        del getitem_192
        buf422 = buf421[0]
        buf423 = buf421[1]
        buf424 = buf421[2]
        del buf421
        buf425 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (512, 768), (768, 1), 0), permute_459, out=buf425)
        del permute_459
        buf426 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (768, 512), (1, 768), 0), view_44, out=buf426)
        buf427 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf424, buf427, 3072, 128, grid=grid(3072), stream=stream0)
        buf428 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf427, buf428, 768, 4, grid=grid(768), stream=stream0)
        buf429 = reinterpret_tensor(buf424, (512, 768), (768, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 768), (768, 1), 0), permute_464, out=buf429)
        del permute_464
        buf430 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (768, 512), (1, 768), 0), view_44, out=buf430)
        buf431 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf423, buf431, 3072, 128, grid=grid(3072), stream=stream0)
        buf432 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf431, buf432, 768, 4, grid=grid(768), stream=stream0)
        buf433 = reinterpret_tensor(buf423, (512, 768), (768, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (512, 768), (768, 1), 0), permute_468, out=buf433)
        del permute_468
        buf434 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_44, out=buf434)
        del view_44
        buf435 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf422, buf435, 3072, 128, grid=grid(3072), stream=stream0)
        buf436 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf435, buf436, 768, 4, grid=grid(768), stream=stream0)
        buf440 = reinterpret_tensor(buf422, (1, 512, 768), (393216, 768, 1), 0); del buf422  # reuse
        buf443 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf413, buf425, buf429, buf433, primals_36, mul_16, div_57, getitem_21, buf440, buf443, 512, 768, grid=grid(512), stream=stream0)
        del div_57
        del getitem_21
        del primals_36
        buf441 = empty((768, ), device='cuda', dtype=torch.float32)
        buf442 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf413, buf425, buf429, buf433, mul_16, buf441, buf442, 768, 512, grid=grid(768), stream=stream0)
        del buf413
        del buf425
        del mul_16
        buf444 = reinterpret_tensor(buf406, (512, 3072), (3072, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (512, 768), (768, 1), 0), permute_472, out=buf444)
        del permute_472
        buf445 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (768, 512), (1, 768), 0), view_42, out=buf445)
        del view_42
        buf446 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf443, buf446, 3072, 128, grid=grid(3072), stream=stream0)
        buf447 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf446, buf447, 768, 4, grid=grid(768), stream=stream0)
        buf448 = reinterpret_tensor(buf444, (1, 512, 3072), (1572864, 3072, 1), 0); del buf444  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf448, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf449 = reinterpret_tensor(buf443, (512, 768), (768, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (512, 3072), (3072, 1), 0), permute_476, out=buf449)
        del permute_476
        buf450 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (3072, 512), (1, 3072), 0), view_40, out=buf450)
        del view_40
        buf451 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf448, buf451, 12288, 128, grid=grid(12288), stream=stream0)
        buf452 = reinterpret_tensor(buf446, (1, 3072), (3072, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf451, buf452, 3072, 4, grid=grid(3072), stream=stream0)
        buf455 = reinterpret_tensor(buf433, (1, 512, 768), (393216, 768, 1), 0); del buf433  # reuse
        buf458 = reinterpret_tensor(buf429, (1, 512, 768), (393216, 768, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf440, buf449, primals_30, mul_11, div_58, getitem_17, buf455, buf458, 512, 768, grid=grid(512), stream=stream0)
        del div_58
        del getitem_17
        del primals_30
        buf456 = empty((768, ), device='cuda', dtype=torch.float32)
        buf457 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf440, buf449, mul_11, buf456, buf457, 768, 512, grid=grid(768), stream=stream0)
        del buf440
        del mul_11
        buf459 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (512, 768), (768, 1), 0), permute_480, out=buf459)
        del permute_480
        buf460 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (768, 512), (1, 768), 0), view_38, out=buf460)
        del view_38
        buf461 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf458, buf461, 3072, 128, grid=grid(3072), stream=stream0)
        buf462 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf461, buf462, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf463 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf459, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_197
        del getitem_198
        del getitem_199
        buf464 = buf463[0]
        buf465 = buf463[1]
        buf466 = buf463[2]
        del buf463
        buf467 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (512, 768), (768, 1), 0), permute_492, out=buf467)
        del permute_492
        buf468 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (768, 512), (1, 768), 0), view_22, out=buf468)
        buf469 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf466, buf469, 3072, 128, grid=grid(3072), stream=stream0)
        buf470 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf469, buf470, 768, 4, grid=grid(768), stream=stream0)
        buf471 = reinterpret_tensor(buf466, (512, 768), (768, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 768), (768, 1), 0), permute_497, out=buf471)
        del permute_497
        buf472 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (768, 512), (1, 768), 0), view_22, out=buf472)
        buf473 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf465, buf473, 3072, 128, grid=grid(3072), stream=stream0)
        buf474 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf473, buf474, 768, 4, grid=grid(768), stream=stream0)
        buf475 = reinterpret_tensor(buf465, (512, 768), (768, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 768), (768, 1), 0), permute_501, out=buf475)
        del permute_501
        buf476 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (768, 512), (1, 768), 0), view_22, out=buf476)
        del view_22
        buf477 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf464, buf477, 3072, 128, grid=grid(3072), stream=stream0)
        buf478 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf477, buf478, 768, 4, grid=grid(768), stream=stream0)
        buf482 = reinterpret_tensor(buf464, (1, 512, 768), (393216, 768, 1), 0); del buf464  # reuse
        buf485 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf455, buf467, buf471, buf475, primals_20, mul_9, div_60, getitem_11, buf482, buf485, 512, 768, grid=grid(512), stream=stream0)
        del div_60
        del getitem_11
        del primals_20
        buf483 = empty((768, ), device='cuda', dtype=torch.float32)
        buf484 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf455, buf467, buf471, buf475, mul_9, buf483, buf484, 768, 512, grid=grid(768), stream=stream0)
        del buf455
        del buf467
        del mul_9
        buf486 = reinterpret_tensor(buf448, (512, 3072), (3072, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (512, 768), (768, 1), 0), permute_505, out=buf486)
        del permute_505
        buf487 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (768, 512), (1, 768), 0), view_20, out=buf487)
        del view_20
        buf488 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf485, buf488, 3072, 128, grid=grid(3072), stream=stream0)
        buf489 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf488, buf489, 768, 4, grid=grid(768), stream=stream0)
        buf490 = reinterpret_tensor(buf486, (1, 512, 3072), (1572864, 3072, 1), 0); del buf486  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf490, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf491 = reinterpret_tensor(buf485, (512, 768), (768, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (512, 3072), (3072, 1), 0), permute_509, out=buf491)
        del permute_509
        buf492 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (3072, 512), (1, 3072), 0), view_18, out=buf492)
        del view_18
        buf493 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf490, buf493, 12288, 128, grid=grid(12288), stream=stream0)
        del buf490
        buf494 = reinterpret_tensor(buf488, (1, 3072), (3072, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf493, buf494, 3072, 4, grid=grid(3072), stream=stream0)
        del buf493
        buf497 = reinterpret_tensor(buf475, (1, 512, 768), (393216, 768, 1), 0); del buf475  # reuse
        buf500 = reinterpret_tensor(buf471, (1, 512, 768), (393216, 768, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf482, buf491, primals_14, mul_4, div_61, getitem_7, buf497, buf500, 512, 768, grid=grid(512), stream=stream0)
        del div_61
        del getitem_7
        del primals_14
        buf498 = empty((768, ), device='cuda', dtype=torch.float32)
        buf499 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf482, buf491, mul_4, buf498, buf499, 768, 512, grid=grid(768), stream=stream0)
        del mul_4
        buf501 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (512, 768), (768, 1), 0), permute_513, out=buf501)
        del permute_513
        buf502 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (768, 512), (1, 768), 0), view_16, out=buf502)
        del view_16
        buf503 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf500, buf503, 3072, 128, grid=grid(3072), stream=stream0)
        buf504 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf503, buf504, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf505 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf501, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_204
        del getitem_205
        del getitem_206
        buf506 = buf505[0]
        buf507 = buf505[1]
        buf508 = buf505[2]
        del buf505
        buf509 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (512, 768), (768, 1), 0), permute_525, out=buf509)
        del permute_525
        buf510 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (768, 512), (1, 768), 0), view, out=buf510)
        buf511 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf508, buf511, 3072, 128, grid=grid(3072), stream=stream0)
        buf512 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf511, buf512, 768, 4, grid=grid(768), stream=stream0)
        buf513 = reinterpret_tensor(buf508, (512, 768), (768, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 768), (768, 1), 0), permute_530, out=buf513)
        del permute_530
        buf514 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (768, 512), (1, 768), 0), view, out=buf514)
        buf515 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf507, buf515, 3072, 128, grid=grid(3072), stream=stream0)
        buf516 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf515, buf516, 768, 4, grid=grid(768), stream=stream0)
        buf517 = reinterpret_tensor(buf507, (512, 768), (768, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (512, 768), (768, 1), 0), permute_534, out=buf517)
        del permute_534
        buf518 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (768, 512), (1, 768), 0), view, out=buf518)
        del view
        buf519 = buf515; del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf506, buf519, 3072, 128, grid=grid(3072), stream=stream0)
        buf520 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf519, buf520, 768, 4, grid=grid(768), stream=stream0)
        del buf519
        buf521 = buf497; del buf497  # reuse
        buf528 = reinterpret_tensor(buf506, (1, 512, 768), (393216, 768, 1), 0); del buf506  # reuse
        buf532 = buf500; del buf500  # reuse
        buf536 = buf482; del buf482  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16.run(buf521, buf509, buf513, buf517, getitem_3, primals_4, mul_2, div_63, add_1, expand, primals_205, buf528, buf532, buf536, 512, 768, grid=grid(512), stream=stream0)
        del buf509
        del buf513
        del buf517
        del div_63
        del getitem_3
        del primals_4
        buf525 = empty((768, ), device='cuda', dtype=torch.float32)
        buf526 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf521, mul_2, buf525, buf526, 768, 512, grid=grid(768), stream=stream0)
        del buf521
        del mul_2
        buf527 = empty((514, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf527, 394752, grid=grid(394752), stream=stream0)
        aten.index_put_(buf527, [add_1], buf528, True)
        del add_1
        del buf528
        buf531 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf531, 768, grid=grid(768), stream=stream0)
        aten.index_put_(buf531, [expand], buf532, True)
        del buf532
        del expand
        buf535 = empty((32005, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_19.run(buf535, 24579840, grid=grid(24579840), stream=stream0)
        aten.index_put_(buf535, [primals_205], buf536, True)
        del buf536
        del primals_205
        return (buf535, buf531, buf527, buf525, buf526, reinterpret_tensor(buf518, (768, 768), (768, 1), 0), reinterpret_tensor(buf520, (768, ), (1, ), 0), reinterpret_tensor(buf514, (768, 768), (768, 1), 0), reinterpret_tensor(buf516, (768, ), (1, ), 0), reinterpret_tensor(buf510, (768, 768), (768, 1), 0), reinterpret_tensor(buf512, (768, ), (1, ), 0), reinterpret_tensor(buf502, (768, 768), (768, 1), 0), reinterpret_tensor(buf504, (768, ), (1, ), 0), buf498, buf499, reinterpret_tensor(buf492, (3072, 768), (768, 1), 0), reinterpret_tensor(buf494, (3072, ), (1, ), 0), reinterpret_tensor(buf487, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf489, (768, ), (1, ), 0), buf483, buf484, reinterpret_tensor(buf476, (768, 768), (768, 1), 0), reinterpret_tensor(buf478, (768, ), (1, ), 0), reinterpret_tensor(buf472, (768, 768), (768, 1), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), reinterpret_tensor(buf468, (768, 768), (768, 1), 0), reinterpret_tensor(buf470, (768, ), (1, ), 0), reinterpret_tensor(buf460, (768, 768), (768, 1), 0), reinterpret_tensor(buf462, (768, ), (1, ), 0), buf456, buf457, reinterpret_tensor(buf450, (3072, 768), (768, 1), 0), reinterpret_tensor(buf452, (3072, ), (1, ), 0), reinterpret_tensor(buf445, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf447, (768, ), (1, ), 0), buf441, buf442, reinterpret_tensor(buf434, (768, 768), (768, 1), 0), reinterpret_tensor(buf436, (768, ), (1, ), 0), reinterpret_tensor(buf430, (768, 768), (768, 1), 0), reinterpret_tensor(buf432, (768, ), (1, ), 0), reinterpret_tensor(buf426, (768, 768), (768, 1), 0), reinterpret_tensor(buf428, (768, ), (1, ), 0), reinterpret_tensor(buf418, (768, 768), (768, 1), 0), reinterpret_tensor(buf420, (768, ), (1, ), 0), buf414, buf415, reinterpret_tensor(buf408, (3072, 768), (768, 1), 0), reinterpret_tensor(buf410, (3072, ), (1, ), 0), reinterpret_tensor(buf403, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf405, (768, ), (1, ), 0), buf399, buf400, reinterpret_tensor(buf392, (768, 768), (768, 1), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), reinterpret_tensor(buf388, (768, 768), (768, 1), 0), reinterpret_tensor(buf390, (768, ), (1, ), 0), reinterpret_tensor(buf384, (768, 768), (768, 1), 0), reinterpret_tensor(buf386, (768, ), (1, ), 0), reinterpret_tensor(buf376, (768, 768), (768, 1), 0), reinterpret_tensor(buf378, (768, ), (1, ), 0), buf372, buf373, reinterpret_tensor(buf366, (3072, 768), (768, 1), 0), reinterpret_tensor(buf368, (3072, ), (1, ), 0), reinterpret_tensor(buf361, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf363, (768, ), (1, ), 0), buf357, buf358, reinterpret_tensor(buf350, (768, 768), (768, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), reinterpret_tensor(buf346, (768, 768), (768, 1), 0), reinterpret_tensor(buf348, (768, ), (1, ), 0), reinterpret_tensor(buf342, (768, 768), (768, 1), 0), reinterpret_tensor(buf344, (768, ), (1, ), 0), reinterpret_tensor(buf334, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (768, ), (1, ), 0), buf330, buf331, reinterpret_tensor(buf324, (3072, 768), (768, 1), 0), reinterpret_tensor(buf326, (3072, ), (1, ), 0), reinterpret_tensor(buf319, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf321, (768, ), (1, ), 0), buf315, buf316, reinterpret_tensor(buf308, (768, 768), (768, 1), 0), reinterpret_tensor(buf310, (768, ), (1, ), 0), reinterpret_tensor(buf304, (768, 768), (768, 1), 0), reinterpret_tensor(buf306, (768, ), (1, ), 0), reinterpret_tensor(buf300, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), reinterpret_tensor(buf292, (768, 768), (768, 1), 0), reinterpret_tensor(buf294, (768, ), (1, ), 0), buf288, buf289, reinterpret_tensor(buf282, (3072, 768), (768, 1), 0), reinterpret_tensor(buf284, (3072, ), (1, ), 0), reinterpret_tensor(buf277, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf279, (768, ), (1, ), 0), buf273, buf274, reinterpret_tensor(buf266, (768, 768), (768, 1), 0), reinterpret_tensor(buf268, (768, ), (1, ), 0), reinterpret_tensor(buf262, (768, 768), (768, 1), 0), reinterpret_tensor(buf264, (768, ), (1, ), 0), reinterpret_tensor(buf258, (768, 768), (768, 1), 0), reinterpret_tensor(buf260, (768, ), (1, ), 0), reinterpret_tensor(buf250, (768, 768), (768, 1), 0), reinterpret_tensor(buf252, (768, ), (1, ), 0), buf246, buf247, reinterpret_tensor(buf240, (3072, 768), (768, 1), 0), reinterpret_tensor(buf242, (3072, ), (1, ), 0), reinterpret_tensor(buf235, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf237, (768, ), (1, ), 0), buf231, buf232, reinterpret_tensor(buf224, (768, 768), (768, 1), 0), reinterpret_tensor(buf226, (768, ), (1, ), 0), reinterpret_tensor(buf220, (768, 768), (768, 1), 0), reinterpret_tensor(buf222, (768, ), (1, ), 0), reinterpret_tensor(buf216, (768, 768), (768, 1), 0), reinterpret_tensor(buf218, (768, ), (1, ), 0), reinterpret_tensor(buf208, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), buf204, buf205, reinterpret_tensor(buf198, (3072, 768), (768, 1), 0), reinterpret_tensor(buf200, (3072, ), (1, ), 0), reinterpret_tensor(buf193, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf195, (768, ), (1, ), 0), buf189, buf190, reinterpret_tensor(buf182, (768, 768), (768, 1), 0), reinterpret_tensor(buf184, (768, ), (1, ), 0), reinterpret_tensor(buf178, (768, 768), (768, 1), 0), reinterpret_tensor(buf180, (768, ), (1, ), 0), reinterpret_tensor(buf174, (768, 768), (768, 1), 0), reinterpret_tensor(buf176, (768, ), (1, ), 0), reinterpret_tensor(buf166, (768, 768), (768, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), buf162, buf163, reinterpret_tensor(buf156, (3072, 768), (768, 1), 0), reinterpret_tensor(buf158, (3072, ), (1, ), 0), reinterpret_tensor(buf151, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), buf147, buf148, reinterpret_tensor(buf140, (768, 768), (768, 1), 0), reinterpret_tensor(buf142, (768, ), (1, ), 0), reinterpret_tensor(buf136, (768, 768), (768, 1), 0), reinterpret_tensor(buf138, (768, ), (1, ), 0), reinterpret_tensor(buf132, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), reinterpret_tensor(buf124, (768, 768), (768, 1), 0), reinterpret_tensor(buf126, (768, ), (1, ), 0), buf120, buf121, reinterpret_tensor(buf114, (3072, 768), (768, 1), 0), reinterpret_tensor(buf116, (3072, ), (1, ), 0), reinterpret_tensor(buf109, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf111, (768, ), (1, ), 0), buf105, buf106, reinterpret_tensor(buf98, (768, 768), (768, 1), 0), reinterpret_tensor(buf100, (768, ), (1, ), 0), reinterpret_tensor(buf94, (768, 768), (768, 1), 0), reinterpret_tensor(buf96, (768, ), (1, ), 0), reinterpret_tensor(buf90, (768, 768), (768, 1), 0), reinterpret_tensor(buf92, (768, ), (1, ), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), buf78, buf79, reinterpret_tensor(buf72, (3072, 768), (768, 1), 0), reinterpret_tensor(buf74, (3072, ), (1, ), 0), reinterpret_tensor(buf67, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf69, (768, ), (1, ), 0), buf63, buf64, reinterpret_tensor(buf56, (768, 768), (768, 1), 0), reinterpret_tensor(buf58, (768, ), (1, ), 0), reinterpret_tensor(buf52, (768, 768), (768, 1), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf50, (768, ), (1, ), 0), reinterpret_tensor(buf40, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), buf36, buf37, reinterpret_tensor(buf30, (3072, 768), (768, 1), 0), reinterpret_tensor(buf32, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf27, (768, ), (1, ), 0), buf21, buf22, reinterpret_tensor(buf15, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf11, buf12, reinterpret_tensor(buf7, (32005, 768), (768, 1), 0), reinterpret_tensor(buf8, (32005, ), (1, ), 0), None, None, None, )


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
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_206 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (514, 1), device='cuda:0', dtype=torch.int64)
    add_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_204 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_190 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_18 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_183 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_30 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_176 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_39 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_46 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_53 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_58 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_148 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_60 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_72 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_74 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_79 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_86 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_72 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_40 = rand_strided((512, 32005), (32005, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((32005, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 32005), (16386560, 32005, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_205, primals_206, expand, add_1, mul_2, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_4, view_18, addmm_4, view_20, getitem_11, mul_9, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_11, view_40, addmm_10, view_42, getitem_21, mul_16, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_18, view_62, addmm_16, view_64, getitem_31, mul_23, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_25, view_84, addmm_22, view_86, getitem_41, mul_30, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_32, view_106, addmm_28, view_108, getitem_51, mul_37, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_39, view_128, addmm_34, view_130, getitem_61, mul_44, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_46, view_150, addmm_40, view_152, getitem_71, mul_51, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_53, view_172, addmm_46, view_174, getitem_81, mul_58, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_60, view_194, addmm_52, view_196, getitem_91, mul_65, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_67, view_216, addmm_58, view_218, getitem_101, mul_72, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_74, view_238, addmm_64, view_240, getitem_111, mul_79, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_81, view_260, addmm_70, view_262, getitem_121, mul_86, view_264, addmm_72, mul_91, view_266, sub_40, convert_element_type_3, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('CamemBert', benchmark_compiled_module)
