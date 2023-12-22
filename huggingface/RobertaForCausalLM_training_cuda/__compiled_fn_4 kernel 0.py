
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


# kernel path: /tmp/torchinductor_youkaichao/43/c43hjw4wwy26gdzr34fyomykxvzksw3kq37b4mk4lavx6yyplwxb.py
# Source Nodes: [], Original ATen: [aten.nll_loss_backward]

triton_poi_fused_nll_loss_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25685415
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


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fvniqkp5ihm2oysgofgtr44fmixufy6oqeb2qqubtflwezscsv.py
# Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss => full_default_2
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 511
    rnumel = 50265
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp3 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp1, tmp6, tmp7)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6bj6yetpxiouqmk33gmy5l7bnpxkfzrftahtm6usironaajdocq.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25735680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 50265)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x1), tmp3 & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp10 = tmp7 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tl.load(in_ptr5 + (x2), tmp3 & xmask, other=0.0)
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.load(in_ptr6 + (x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp13 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp3, tmp18, tmp19)
    tmp21 = tl.where(tmp3, tmp20, tmp11)
    tmp22 = tmp0 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz25inbd7pplbr67prd5pllqe56csj56ho4u5sxlslihmfdbdgz2.py
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50265
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
        tmp0 = tl.load(in_ptr0 + (x0 + (50265*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kl/ckl64g4qlw6lxydee3zwgudsyyo67d7jb6iocsmgv3pw4ri4uhqq.py
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
    tmp32 = tl.full([1], 0, tl.int64)
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


# kernel path: /tmp/torchinductor_youkaichao/o6/co6kginoi4hbnpwb3e2ohsqck3tmtl3hnudfazjvvmmmno3e5sr3.py
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
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5mifkq5gfsp2tmqdkbfbhjvyvsvvjhbo5ceieakvgmvfidr4hp.py
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
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ax/caxn7gmqkfjnhtrpeoryh7v5ssaysjc2iafirdoxygvtckav74cm.py
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
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_19', 'mutated_arg_names': []},
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, add_1, mul_2, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_4, view_18, addmm_4, view_20, getitem_11, mul_9, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_11, view_40, addmm_10, view_42, getitem_21, mul_16, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_18, view_62, addmm_16, view_64, getitem_31, mul_23, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_25, view_84, addmm_22, view_86, getitem_41, mul_30, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_32, view_106, addmm_28, view_108, getitem_51, mul_37, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_39, view_128, addmm_34, view_130, getitem_61, mul_44, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_46, view_150, addmm_40, view_152, getitem_71, mul_51, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_53, view_172, addmm_46, view_174, getitem_81, mul_58, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_60, view_194, addmm_52, view_196, getitem_91, mul_65, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_67, view_216, addmm_58, view_218, getitem_101, mul_72, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_74, view_238, addmm_64, view_240, getitem_111, mul_79, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_81, view_260, addmm_70, view_262, getitem_121, mul_86, view_264, addmm_72, mul_91, view_266, sub_40, convert_element_type_3, ne_4, where_2, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2 = args
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
    assert_size_stride(primals_206, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
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
    assert_size_stride(sub_40, (511, 50265), (50265, 1))
    assert_size_stride(convert_element_type_3, (), ())
    assert_size_stride(ne_4, (511, 1), (1, 1))
    assert_size_stride(where_2, (511, 1), (1, 1))
    assert_size_stride(permute_134, (50265, 768), (768, 1))
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
    assert_size_stride(tangents_2, (1, 512, 50265), (25735680, 50265, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((511, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_0.run(buf0, 25685415, grid=grid(25685415), stream=stream0)
        aten.scatter_(buf0,1,where_2,-1.0)
        del where_2
        buf3 = empty_strided((511, 1), (1, 511), device='cuda', dtype=torch.float32)
        # Source Nodes: [loss], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1.run(buf0, ne_4, tangents_1, convert_element_type_3, buf3, 511, 50265, grid=grid(511), stream=stream0)
        buf4 = empty((1, 512, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_2.run(tangents_2, buf0, ne_4, tangents_1, convert_element_type_3, sub_40, buf3, buf4, 25735680, grid=grid(25735680), stream=stream0)
        del buf0
        del buf3
        del convert_element_type_3
        del ne_4
        del sub_40
        del tangents_1
        del tangents_2
        buf5 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 50265), (50265, 1), 0), permute_134, out=buf5)
        del permute_134
        buf6 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (50265, 512), (1, 50265), 0), view_266, out=buf6)
        del view_266
        buf7 = empty((1, 50265), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf7, 50265, 512, grid=grid(50265), stream=stream0)
        del buf4
        buf12 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_4.run(buf5, primals_200, mul_91, div_26, addmm_72, buf12, 512, 768, grid=grid(512), stream=stream0)
        del addmm_72
        del div_26
        del primals_200
        buf10 = empty((768, ), device='cuda', dtype=torch.float32)
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf5, mul_91, buf10, buf11, 768, 512, grid=grid(768), stream=stream0)
        del mul_91
        buf13 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 768), (768, 1), 0), permute_138, out=buf13)
        del permute_138
        buf14 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (768, 512), (1, 768), 0), view_264, out=buf14)
        del view_264
        buf15 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf12, buf15, 3072, 128, grid=grid(3072), stream=stream0)
        buf16 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf15, buf16, 768, 4, grid=grid(768), stream=stream0)
        buf19 = buf12; del buf12  # reuse
        buf22 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_8.run(buf13, primals_196, mul_86, div_27, getitem_121, buf19, buf22, 512, 768, grid=grid(512), stream=stream0)
        del div_27
        del getitem_121
        del primals_196
        buf20 = empty((768, ), device='cuda', dtype=torch.float32)
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf13, mul_86, buf20, buf21, 768, 512, grid=grid(768), stream=stream0)
        del mul_86
        buf23 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 768), (768, 1), 0), permute_142, out=buf23)
        del permute_142
        buf24 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (768, 512), (1, 768), 0), view_262, out=buf24)
        del view_262
        buf25 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf22, buf25, 3072, 128, grid=grid(3072), stream=stream0)
        buf26 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf25, buf26, 768, 4, grid=grid(768), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (1, 512, 3072), (1572864, 3072, 1), 0); del buf23  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf27, addmm_70, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_70
        buf28 = reinterpret_tensor(buf22, (512, 768), (768, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0), permute_146, out=buf28)
        del permute_146
        buf29 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (3072, 512), (1, 3072), 0), view_260, out=buf29)
        del view_260
        buf30 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf27, buf30, 12288, 128, grid=grid(12288), stream=stream0)
        buf31 = reinterpret_tensor(buf25, (1, 3072), (3072, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf30, buf31, 3072, 4, grid=grid(3072), stream=stream0)
        buf34 = reinterpret_tensor(buf13, (1, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
        buf37 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf19, buf28, primals_190, mul_81, div_28, getitem_117, buf34, buf37, 512, 768, grid=grid(512), stream=stream0)
        del div_28
        del getitem_117
        del primals_190
        buf35 = empty((768, ), device='cuda', dtype=torch.float32)
        buf36 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf19, buf28, mul_81, buf35, buf36, 768, 512, grid=grid(768), stream=stream0)
        del buf19
        del mul_81
        buf38 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (512, 768), (768, 1), 0), permute_150, out=buf38)
        del permute_150
        buf39 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (768, 512), (1, 768), 0), view_258, out=buf39)
        del view_258
        buf40 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf37, buf40, 3072, 128, grid=grid(3072), stream=stream0)
        buf41 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf40, buf41, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf42 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf38, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_127
        del getitem_128
        del getitem_129
        buf43 = buf42[0]
        buf44 = buf42[1]
        buf45 = buf42[2]
        del buf42
        buf46 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (512, 768), (768, 1), 0), permute_162, out=buf46)
        del permute_162
        buf47 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (768, 512), (1, 768), 0), view_242, out=buf47)
        buf48 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf45, buf48, 3072, 128, grid=grid(3072), stream=stream0)
        buf49 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf48, buf49, 768, 4, grid=grid(768), stream=stream0)
        buf50 = reinterpret_tensor(buf45, (512, 768), (768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 768), (768, 1), 0), permute_167, out=buf50)
        del permute_167
        buf51 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (768, 512), (1, 768), 0), view_242, out=buf51)
        buf52 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf44, buf52, 3072, 128, grid=grid(3072), stream=stream0)
        buf53 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf52, buf53, 768, 4, grid=grid(768), stream=stream0)
        buf54 = reinterpret_tensor(buf44, (512, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (512, 768), (768, 1), 0), permute_171, out=buf54)
        del permute_171
        buf55 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (768, 512), (1, 768), 0), view_242, out=buf55)
        del view_242
        buf56 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf43, buf56, 3072, 128, grid=grid(3072), stream=stream0)
        buf57 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf56, buf57, 768, 4, grid=grid(768), stream=stream0)
        buf61 = reinterpret_tensor(buf43, (1, 512, 768), (393216, 768, 1), 0); del buf43  # reuse
        buf64 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf34, buf46, buf50, buf54, primals_180, mul_79, div_30, getitem_111, buf61, buf64, 512, 768, grid=grid(512), stream=stream0)
        del div_30
        del getitem_111
        del primals_180
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        buf63 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf34, buf46, buf50, buf54, mul_79, buf62, buf63, 768, 512, grid=grid(768), stream=stream0)
        del buf34
        del buf46
        del mul_79
        buf65 = reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 768), (768, 1), 0), permute_175, out=buf65)
        del permute_175
        buf66 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (768, 512), (1, 768), 0), view_240, out=buf66)
        del view_240
        buf67 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf64, buf67, 3072, 128, grid=grid(3072), stream=stream0)
        buf68 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf67, buf68, 768, 4, grid=grid(768), stream=stream0)
        buf69 = reinterpret_tensor(buf65, (1, 512, 3072), (1572864, 3072, 1), 0); del buf65  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf69, addmm_64, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_64
        buf70 = reinterpret_tensor(buf64, (512, 768), (768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), permute_179, out=buf70)
        del permute_179
        buf71 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (3072, 512), (1, 3072), 0), view_238, out=buf71)
        del view_238
        buf72 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf69, buf72, 12288, 128, grid=grid(12288), stream=stream0)
        buf73 = reinterpret_tensor(buf67, (1, 3072), (3072, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf72, buf73, 3072, 4, grid=grid(3072), stream=stream0)
        buf76 = reinterpret_tensor(buf54, (1, 512, 768), (393216, 768, 1), 0); del buf54  # reuse
        buf79 = reinterpret_tensor(buf50, (1, 512, 768), (393216, 768, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf61, buf70, primals_174, mul_74, div_31, getitem_107, buf76, buf79, 512, 768, grid=grid(512), stream=stream0)
        del div_31
        del getitem_107
        del primals_174
        buf77 = empty((768, ), device='cuda', dtype=torch.float32)
        buf78 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf61, buf70, mul_74, buf77, buf78, 768, 512, grid=grid(768), stream=stream0)
        del buf61
        del mul_74
        buf80 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (512, 768), (768, 1), 0), permute_183, out=buf80)
        del permute_183
        buf81 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (768, 512), (1, 768), 0), view_236, out=buf81)
        del view_236
        buf82 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf79, buf82, 3072, 128, grid=grid(3072), stream=stream0)
        buf83 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf82, buf83, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf84 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf80, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_134
        del getitem_135
        del getitem_136
        buf85 = buf84[0]
        buf86 = buf84[1]
        buf87 = buf84[2]
        del buf84
        buf88 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 768), (768, 1), 0), permute_195, out=buf88)
        del permute_195
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (768, 512), (1, 768), 0), view_220, out=buf89)
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf87, buf90, 3072, 128, grid=grid(3072), stream=stream0)
        buf91 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf90, buf91, 768, 4, grid=grid(768), stream=stream0)
        buf92 = reinterpret_tensor(buf87, (512, 768), (768, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (512, 768), (768, 1), 0), permute_200, out=buf92)
        del permute_200
        buf93 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (768, 512), (1, 768), 0), view_220, out=buf93)
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf86, buf94, 3072, 128, grid=grid(3072), stream=stream0)
        buf95 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf94, buf95, 768, 4, grid=grid(768), stream=stream0)
        buf96 = reinterpret_tensor(buf86, (512, 768), (768, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (512, 768), (768, 1), 0), permute_204, out=buf96)
        del permute_204
        buf97 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (768, 512), (1, 768), 0), view_220, out=buf97)
        del view_220
        buf98 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf85, buf98, 3072, 128, grid=grid(3072), stream=stream0)
        buf99 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf98, buf99, 768, 4, grid=grid(768), stream=stream0)
        buf103 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
        buf106 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf76, buf88, buf92, buf96, primals_164, mul_72, div_33, getitem_101, buf103, buf106, 512, 768, grid=grid(512), stream=stream0)
        del div_33
        del getitem_101
        del primals_164
        buf104 = empty((768, ), device='cuda', dtype=torch.float32)
        buf105 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf76, buf88, buf92, buf96, mul_72, buf104, buf105, 768, 512, grid=grid(768), stream=stream0)
        del buf76
        del buf88
        del mul_72
        buf107 = reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (512, 768), (768, 1), 0), permute_208, out=buf107)
        del permute_208
        buf108 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (768, 512), (1, 768), 0), view_218, out=buf108)
        del view_218
        buf109 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf106, buf109, 3072, 128, grid=grid(3072), stream=stream0)
        buf110 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf109, buf110, 768, 4, grid=grid(768), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (1, 512, 3072), (1572864, 3072, 1), 0); del buf107  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf111, addmm_58, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_58
        buf112 = reinterpret_tensor(buf106, (512, 768), (768, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (512, 3072), (3072, 1), 0), permute_212, out=buf112)
        del permute_212
        buf113 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (3072, 512), (1, 3072), 0), view_216, out=buf113)
        del view_216
        buf114 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf111, buf114, 12288, 128, grid=grid(12288), stream=stream0)
        buf115 = reinterpret_tensor(buf109, (1, 3072), (3072, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf114, buf115, 3072, 4, grid=grid(3072), stream=stream0)
        buf118 = reinterpret_tensor(buf96, (1, 512, 768), (393216, 768, 1), 0); del buf96  # reuse
        buf121 = reinterpret_tensor(buf92, (1, 512, 768), (393216, 768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf103, buf112, primals_158, mul_67, div_34, getitem_97, buf118, buf121, 512, 768, grid=grid(512), stream=stream0)
        del div_34
        del getitem_97
        del primals_158
        buf119 = empty((768, ), device='cuda', dtype=torch.float32)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf103, buf112, mul_67, buf119, buf120, 768, 512, grid=grid(768), stream=stream0)
        del buf103
        del mul_67
        buf122 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 768), (768, 1), 0), permute_216, out=buf122)
        del permute_216
        buf123 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (768, 512), (1, 768), 0), view_214, out=buf123)
        del view_214
        buf124 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf121, buf124, 3072, 128, grid=grid(3072), stream=stream0)
        buf125 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf124, buf125, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf126 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf122, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_141
        del getitem_142
        del getitem_143
        buf127 = buf126[0]
        buf128 = buf126[1]
        buf129 = buf126[2]
        del buf126
        buf130 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (512, 768), (768, 1), 0), permute_228, out=buf130)
        del permute_228
        buf131 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (768, 512), (1, 768), 0), view_198, out=buf131)
        buf132 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf129, buf132, 3072, 128, grid=grid(3072), stream=stream0)
        buf133 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf132, buf133, 768, 4, grid=grid(768), stream=stream0)
        buf134 = reinterpret_tensor(buf129, (512, 768), (768, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 768), (768, 1), 0), permute_233, out=buf134)
        del permute_233
        buf135 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (768, 512), (1, 768), 0), view_198, out=buf135)
        buf136 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf128, buf136, 3072, 128, grid=grid(3072), stream=stream0)
        buf137 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf136, buf137, 768, 4, grid=grid(768), stream=stream0)
        buf138 = reinterpret_tensor(buf128, (512, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (512, 768), (768, 1), 0), permute_237, out=buf138)
        del permute_237
        buf139 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (768, 512), (1, 768), 0), view_198, out=buf139)
        del view_198
        buf140 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf127, buf140, 3072, 128, grid=grid(3072), stream=stream0)
        buf141 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf140, buf141, 768, 4, grid=grid(768), stream=stream0)
        buf145 = reinterpret_tensor(buf127, (1, 512, 768), (393216, 768, 1), 0); del buf127  # reuse
        buf148 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf118, buf130, buf134, buf138, primals_148, mul_65, div_36, getitem_91, buf145, buf148, 512, 768, grid=grid(512), stream=stream0)
        del div_36
        del getitem_91
        del primals_148
        buf146 = empty((768, ), device='cuda', dtype=torch.float32)
        buf147 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf118, buf130, buf134, buf138, mul_65, buf146, buf147, 768, 512, grid=grid(768), stream=stream0)
        del buf118
        del buf130
        del mul_65
        buf149 = reinterpret_tensor(buf111, (512, 3072), (3072, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 768), (768, 1), 0), permute_241, out=buf149)
        del permute_241
        buf150 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (768, 512), (1, 768), 0), view_196, out=buf150)
        del view_196
        buf151 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf148, buf151, 3072, 128, grid=grid(3072), stream=stream0)
        buf152 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf151, buf152, 768, 4, grid=grid(768), stream=stream0)
        buf153 = reinterpret_tensor(buf149, (1, 512, 3072), (1572864, 3072, 1), 0); del buf149  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf153, addmm_52, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_52
        buf154 = reinterpret_tensor(buf148, (512, 768), (768, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 3072), (3072, 1), 0), permute_245, out=buf154)
        del permute_245
        buf155 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (3072, 512), (1, 3072), 0), view_194, out=buf155)
        del view_194
        buf156 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf153, buf156, 12288, 128, grid=grid(12288), stream=stream0)
        buf157 = reinterpret_tensor(buf151, (1, 3072), (3072, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf156, buf157, 3072, 4, grid=grid(3072), stream=stream0)
        buf160 = reinterpret_tensor(buf138, (1, 512, 768), (393216, 768, 1), 0); del buf138  # reuse
        buf163 = reinterpret_tensor(buf134, (1, 512, 768), (393216, 768, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf145, buf154, primals_142, mul_60, div_37, getitem_87, buf160, buf163, 512, 768, grid=grid(512), stream=stream0)
        del div_37
        del getitem_87
        del primals_142
        buf161 = empty((768, ), device='cuda', dtype=torch.float32)
        buf162 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf145, buf154, mul_60, buf161, buf162, 768, 512, grid=grid(768), stream=stream0)
        del buf145
        del mul_60
        buf164 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (512, 768), (768, 1), 0), permute_249, out=buf164)
        del permute_249
        buf165 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (768, 512), (1, 768), 0), view_192, out=buf165)
        del view_192
        buf166 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf163, buf166, 3072, 128, grid=grid(3072), stream=stream0)
        buf167 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf166, buf167, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf168 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf164, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_148
        del getitem_149
        del getitem_150
        buf169 = buf168[0]
        buf170 = buf168[1]
        buf171 = buf168[2]
        del buf168
        buf172 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 768), (768, 1), 0), permute_261, out=buf172)
        del permute_261
        buf173 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (768, 512), (1, 768), 0), view_176, out=buf173)
        buf174 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf171, buf174, 3072, 128, grid=grid(3072), stream=stream0)
        buf175 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf174, buf175, 768, 4, grid=grid(768), stream=stream0)
        buf176 = reinterpret_tensor(buf171, (512, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (512, 768), (768, 1), 0), permute_266, out=buf176)
        del permute_266
        buf177 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf170, (768, 512), (1, 768), 0), view_176, out=buf177)
        buf178 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf170, buf178, 3072, 128, grid=grid(3072), stream=stream0)
        buf179 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf178, buf179, 768, 4, grid=grid(768), stream=stream0)
        buf180 = reinterpret_tensor(buf170, (512, 768), (768, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (512, 768), (768, 1), 0), permute_270, out=buf180)
        del permute_270
        buf181 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (768, 512), (1, 768), 0), view_176, out=buf181)
        del view_176
        buf182 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf169, buf182, 3072, 128, grid=grid(3072), stream=stream0)
        buf183 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf182, buf183, 768, 4, grid=grid(768), stream=stream0)
        buf187 = reinterpret_tensor(buf169, (1, 512, 768), (393216, 768, 1), 0); del buf169  # reuse
        buf190 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf160, buf172, buf176, buf180, primals_132, mul_58, div_39, getitem_81, buf187, buf190, 512, 768, grid=grid(512), stream=stream0)
        del div_39
        del getitem_81
        del primals_132
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        buf189 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf160, buf172, buf176, buf180, mul_58, buf188, buf189, 768, 512, grid=grid(768), stream=stream0)
        del buf160
        del buf172
        del mul_58
        buf191 = reinterpret_tensor(buf153, (512, 3072), (3072, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 768), (768, 1), 0), permute_274, out=buf191)
        del permute_274
        buf192 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (768, 512), (1, 768), 0), view_174, out=buf192)
        del view_174
        buf193 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf190, buf193, 3072, 128, grid=grid(3072), stream=stream0)
        buf194 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf193, buf194, 768, 4, grid=grid(768), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (1, 512, 3072), (1572864, 3072, 1), 0); del buf191  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf195, addmm_46, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_46
        buf196 = reinterpret_tensor(buf190, (512, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 3072), (3072, 1), 0), permute_278, out=buf196)
        del permute_278
        buf197 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (3072, 512), (1, 3072), 0), view_172, out=buf197)
        del view_172
        buf198 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf195, buf198, 12288, 128, grid=grid(12288), stream=stream0)
        buf199 = reinterpret_tensor(buf193, (1, 3072), (3072, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf198, buf199, 3072, 4, grid=grid(3072), stream=stream0)
        buf202 = reinterpret_tensor(buf180, (1, 512, 768), (393216, 768, 1), 0); del buf180  # reuse
        buf205 = reinterpret_tensor(buf176, (1, 512, 768), (393216, 768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf187, buf196, primals_126, mul_53, div_40, getitem_77, buf202, buf205, 512, 768, grid=grid(512), stream=stream0)
        del div_40
        del getitem_77
        del primals_126
        buf203 = empty((768, ), device='cuda', dtype=torch.float32)
        buf204 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf187, buf196, mul_53, buf203, buf204, 768, 512, grid=grid(768), stream=stream0)
        del buf187
        del mul_53
        buf206 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (512, 768), (768, 1), 0), permute_282, out=buf206)
        del permute_282
        buf207 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (768, 512), (1, 768), 0), view_170, out=buf207)
        del view_170
        buf208 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf205, buf208, 3072, 128, grid=grid(3072), stream=stream0)
        buf209 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf208, buf209, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf210 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf206, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_155
        del getitem_156
        del getitem_157
        buf211 = buf210[0]
        buf212 = buf210[1]
        buf213 = buf210[2]
        del buf210
        buf214 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (512, 768), (768, 1), 0), permute_294, out=buf214)
        del permute_294
        buf215 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (768, 512), (1, 768), 0), view_154, out=buf215)
        buf216 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf213, buf216, 3072, 128, grid=grid(3072), stream=stream0)
        buf217 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf216, buf217, 768, 4, grid=grid(768), stream=stream0)
        buf218 = reinterpret_tensor(buf213, (512, 768), (768, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 768), (768, 1), 0), permute_299, out=buf218)
        del permute_299
        buf219 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 512), (1, 768), 0), view_154, out=buf219)
        buf220 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf212, buf220, 3072, 128, grid=grid(3072), stream=stream0)
        buf221 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf220, buf221, 768, 4, grid=grid(768), stream=stream0)
        buf222 = reinterpret_tensor(buf212, (512, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (512, 768), (768, 1), 0), permute_303, out=buf222)
        del permute_303
        buf223 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf211, (768, 512), (1, 768), 0), view_154, out=buf223)
        del view_154
        buf224 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf211, buf224, 3072, 128, grid=grid(3072), stream=stream0)
        buf225 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf224, buf225, 768, 4, grid=grid(768), stream=stream0)
        buf229 = reinterpret_tensor(buf211, (1, 512, 768), (393216, 768, 1), 0); del buf211  # reuse
        buf232 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf202, buf214, buf218, buf222, primals_116, mul_51, div_42, getitem_71, buf229, buf232, 512, 768, grid=grid(512), stream=stream0)
        del div_42
        del getitem_71
        del primals_116
        buf230 = empty((768, ), device='cuda', dtype=torch.float32)
        buf231 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf202, buf214, buf218, buf222, mul_51, buf230, buf231, 768, 512, grid=grid(768), stream=stream0)
        del buf202
        del buf214
        del mul_51
        buf233 = reinterpret_tensor(buf195, (512, 3072), (3072, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 768), (768, 1), 0), permute_307, out=buf233)
        del permute_307
        buf234 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 512), (1, 768), 0), view_152, out=buf234)
        del view_152
        buf235 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf232, buf235, 3072, 128, grid=grid(3072), stream=stream0)
        buf236 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf235, buf236, 768, 4, grid=grid(768), stream=stream0)
        buf237 = reinterpret_tensor(buf233, (1, 512, 3072), (1572864, 3072, 1), 0); del buf233  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf237, addmm_40, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_40
        buf238 = reinterpret_tensor(buf232, (512, 768), (768, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 3072), (3072, 1), 0), permute_311, out=buf238)
        del permute_311
        buf239 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (3072, 512), (1, 3072), 0), view_150, out=buf239)
        del view_150
        buf240 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf237, buf240, 12288, 128, grid=grid(12288), stream=stream0)
        buf241 = reinterpret_tensor(buf235, (1, 3072), (3072, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf240, buf241, 3072, 4, grid=grid(3072), stream=stream0)
        buf244 = reinterpret_tensor(buf222, (1, 512, 768), (393216, 768, 1), 0); del buf222  # reuse
        buf247 = reinterpret_tensor(buf218, (1, 512, 768), (393216, 768, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf229, buf238, primals_110, mul_46, div_43, getitem_67, buf244, buf247, 512, 768, grid=grid(512), stream=stream0)
        del div_43
        del getitem_67
        del primals_110
        buf245 = empty((768, ), device='cuda', dtype=torch.float32)
        buf246 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf229, buf238, mul_46, buf245, buf246, 768, 512, grid=grid(768), stream=stream0)
        del buf229
        del mul_46
        buf248 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (512, 768), (768, 1), 0), permute_315, out=buf248)
        del permute_315
        buf249 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (768, 512), (1, 768), 0), view_148, out=buf249)
        del view_148
        buf250 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf247, buf250, 3072, 128, grid=grid(3072), stream=stream0)
        buf251 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf250, buf251, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf252 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf248, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_162
        del getitem_163
        del getitem_164
        buf253 = buf252[0]
        buf254 = buf252[1]
        buf255 = buf252[2]
        del buf252
        buf256 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 768), (768, 1), 0), permute_327, out=buf256)
        del permute_327
        buf257 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (768, 512), (1, 768), 0), view_132, out=buf257)
        buf258 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf255, buf258, 3072, 128, grid=grid(3072), stream=stream0)
        buf259 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf258, buf259, 768, 4, grid=grid(768), stream=stream0)
        buf260 = reinterpret_tensor(buf255, (512, 768), (768, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (512, 768), (768, 1), 0), permute_332, out=buf260)
        del permute_332
        buf261 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (768, 512), (1, 768), 0), view_132, out=buf261)
        buf262 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf254, buf262, 3072, 128, grid=grid(3072), stream=stream0)
        buf263 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf262, buf263, 768, 4, grid=grid(768), stream=stream0)
        buf264 = reinterpret_tensor(buf254, (512, 768), (768, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 768), (768, 1), 0), permute_336, out=buf264)
        del permute_336
        buf265 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (768, 512), (1, 768), 0), view_132, out=buf265)
        del view_132
        buf266 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf253, buf266, 3072, 128, grid=grid(3072), stream=stream0)
        buf267 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf266, buf267, 768, 4, grid=grid(768), stream=stream0)
        buf271 = reinterpret_tensor(buf253, (1, 512, 768), (393216, 768, 1), 0); del buf253  # reuse
        buf274 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf244, buf256, buf260, buf264, primals_100, mul_44, div_45, getitem_61, buf271, buf274, 512, 768, grid=grid(512), stream=stream0)
        del div_45
        del getitem_61
        del primals_100
        buf272 = empty((768, ), device='cuda', dtype=torch.float32)
        buf273 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf244, buf256, buf260, buf264, mul_44, buf272, buf273, 768, 512, grid=grid(768), stream=stream0)
        del buf244
        del buf256
        del mul_44
        buf275 = reinterpret_tensor(buf237, (512, 3072), (3072, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (512, 768), (768, 1), 0), permute_340, out=buf275)
        del permute_340
        buf276 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (768, 512), (1, 768), 0), view_130, out=buf276)
        del view_130
        buf277 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf274, buf277, 3072, 128, grid=grid(3072), stream=stream0)
        buf278 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf277, buf278, 768, 4, grid=grid(768), stream=stream0)
        buf279 = reinterpret_tensor(buf275, (1, 512, 3072), (1572864, 3072, 1), 0); del buf275  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf279, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf280 = reinterpret_tensor(buf274, (512, 768), (768, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 3072), (3072, 1), 0), permute_344, out=buf280)
        del permute_344
        buf281 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (3072, 512), (1, 3072), 0), view_128, out=buf281)
        del view_128
        buf282 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf279, buf282, 12288, 128, grid=grid(12288), stream=stream0)
        buf283 = reinterpret_tensor(buf277, (1, 3072), (3072, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf282, buf283, 3072, 4, grid=grid(3072), stream=stream0)
        buf286 = reinterpret_tensor(buf264, (1, 512, 768), (393216, 768, 1), 0); del buf264  # reuse
        buf289 = reinterpret_tensor(buf260, (1, 512, 768), (393216, 768, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf271, buf280, primals_94, mul_39, div_46, getitem_57, buf286, buf289, 512, 768, grid=grid(512), stream=stream0)
        del div_46
        del getitem_57
        del primals_94
        buf287 = empty((768, ), device='cuda', dtype=torch.float32)
        buf288 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf271, buf280, mul_39, buf287, buf288, 768, 512, grid=grid(768), stream=stream0)
        del buf271
        del mul_39
        buf290 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (512, 768), (768, 1), 0), permute_348, out=buf290)
        del permute_348
        buf291 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (768, 512), (1, 768), 0), view_126, out=buf291)
        del view_126
        buf292 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf289, buf292, 3072, 128, grid=grid(3072), stream=stream0)
        buf293 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf292, buf293, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf294 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf290, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_169
        del getitem_170
        del getitem_171
        buf295 = buf294[0]
        buf296 = buf294[1]
        buf297 = buf294[2]
        del buf294
        buf298 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (512, 768), (768, 1), 0), permute_360, out=buf298)
        del permute_360
        buf299 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (768, 512), (1, 768), 0), view_110, out=buf299)
        buf300 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf297, buf300, 3072, 128, grid=grid(3072), stream=stream0)
        buf301 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf300, buf301, 768, 4, grid=grid(768), stream=stream0)
        buf302 = reinterpret_tensor(buf297, (512, 768), (768, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 768), (768, 1), 0), permute_365, out=buf302)
        del permute_365
        buf303 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (768, 512), (1, 768), 0), view_110, out=buf303)
        buf304 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf296, buf304, 3072, 128, grid=grid(3072), stream=stream0)
        buf305 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf304, buf305, 768, 4, grid=grid(768), stream=stream0)
        buf306 = reinterpret_tensor(buf296, (512, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (512, 768), (768, 1), 0), permute_369, out=buf306)
        del permute_369
        buf307 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (768, 512), (1, 768), 0), view_110, out=buf307)
        del view_110
        buf308 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf295, buf308, 3072, 128, grid=grid(3072), stream=stream0)
        buf309 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf308, buf309, 768, 4, grid=grid(768), stream=stream0)
        buf313 = reinterpret_tensor(buf295, (1, 512, 768), (393216, 768, 1), 0); del buf295  # reuse
        buf316 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf286, buf298, buf302, buf306, primals_84, mul_37, div_48, getitem_51, buf313, buf316, 512, 768, grid=grid(512), stream=stream0)
        del div_48
        del getitem_51
        del primals_84
        buf314 = empty((768, ), device='cuda', dtype=torch.float32)
        buf315 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf286, buf298, buf302, buf306, mul_37, buf314, buf315, 768, 512, grid=grid(768), stream=stream0)
        del buf286
        del buf298
        del mul_37
        buf317 = reinterpret_tensor(buf279, (512, 3072), (3072, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (512, 768), (768, 1), 0), permute_373, out=buf317)
        del permute_373
        buf318 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (768, 512), (1, 768), 0), view_108, out=buf318)
        del view_108
        buf319 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf316, buf319, 3072, 128, grid=grid(3072), stream=stream0)
        buf320 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf319, buf320, 768, 4, grid=grid(768), stream=stream0)
        buf321 = reinterpret_tensor(buf317, (1, 512, 3072), (1572864, 3072, 1), 0); del buf317  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf321, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf322 = reinterpret_tensor(buf316, (512, 768), (768, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (512, 3072), (3072, 1), 0), permute_377, out=buf322)
        del permute_377
        buf323 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (3072, 512), (1, 3072), 0), view_106, out=buf323)
        del view_106
        buf324 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf321, buf324, 12288, 128, grid=grid(12288), stream=stream0)
        buf325 = reinterpret_tensor(buf319, (1, 3072), (3072, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf324, buf325, 3072, 4, grid=grid(3072), stream=stream0)
        buf328 = reinterpret_tensor(buf306, (1, 512, 768), (393216, 768, 1), 0); del buf306  # reuse
        buf331 = reinterpret_tensor(buf302, (1, 512, 768), (393216, 768, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf313, buf322, primals_78, mul_32, div_49, getitem_47, buf328, buf331, 512, 768, grid=grid(512), stream=stream0)
        del div_49
        del getitem_47
        del primals_78
        buf329 = empty((768, ), device='cuda', dtype=torch.float32)
        buf330 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf313, buf322, mul_32, buf329, buf330, 768, 512, grid=grid(768), stream=stream0)
        del buf313
        del mul_32
        buf332 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (512, 768), (768, 1), 0), permute_381, out=buf332)
        del permute_381
        buf333 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (768, 512), (1, 768), 0), view_104, out=buf333)
        del view_104
        buf334 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf331, buf334, 3072, 128, grid=grid(3072), stream=stream0)
        buf335 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf334, buf335, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf336 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf332, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_176
        del getitem_177
        del getitem_178
        buf337 = buf336[0]
        buf338 = buf336[1]
        buf339 = buf336[2]
        del buf336
        buf340 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 768), (768, 1), 0), permute_393, out=buf340)
        del permute_393
        buf341 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (768, 512), (1, 768), 0), view_88, out=buf341)
        buf342 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf339, buf342, 3072, 128, grid=grid(3072), stream=stream0)
        buf343 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf342, buf343, 768, 4, grid=grid(768), stream=stream0)
        buf344 = reinterpret_tensor(buf339, (512, 768), (768, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (512, 768), (768, 1), 0), permute_398, out=buf344)
        del permute_398
        buf345 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (768, 512), (1, 768), 0), view_88, out=buf345)
        buf346 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf338, buf346, 3072, 128, grid=grid(3072), stream=stream0)
        buf347 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf346, buf347, 768, 4, grid=grid(768), stream=stream0)
        buf348 = reinterpret_tensor(buf338, (512, 768), (768, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (512, 768), (768, 1), 0), permute_402, out=buf348)
        del permute_402
        buf349 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (768, 512), (1, 768), 0), view_88, out=buf349)
        del view_88
        buf350 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf337, buf350, 3072, 128, grid=grid(3072), stream=stream0)
        buf351 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf350, buf351, 768, 4, grid=grid(768), stream=stream0)
        buf355 = reinterpret_tensor(buf337, (1, 512, 768), (393216, 768, 1), 0); del buf337  # reuse
        buf358 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf328, buf340, buf344, buf348, primals_68, mul_30, div_51, getitem_41, buf355, buf358, 512, 768, grid=grid(512), stream=stream0)
        del div_51
        del getitem_41
        del primals_68
        buf356 = empty((768, ), device='cuda', dtype=torch.float32)
        buf357 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf328, buf340, buf344, buf348, mul_30, buf356, buf357, 768, 512, grid=grid(768), stream=stream0)
        del buf328
        del buf340
        del mul_30
        buf359 = reinterpret_tensor(buf321, (512, 3072), (3072, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (512, 768), (768, 1), 0), permute_406, out=buf359)
        del permute_406
        buf360 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (768, 512), (1, 768), 0), view_86, out=buf360)
        del view_86
        buf361 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf358, buf361, 3072, 128, grid=grid(3072), stream=stream0)
        buf362 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf361, buf362, 768, 4, grid=grid(768), stream=stream0)
        buf363 = reinterpret_tensor(buf359, (1, 512, 3072), (1572864, 3072, 1), 0); del buf359  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf363, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf364 = reinterpret_tensor(buf358, (512, 768), (768, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 3072), (3072, 1), 0), permute_410, out=buf364)
        del permute_410
        buf365 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (3072, 512), (1, 3072), 0), view_84, out=buf365)
        del view_84
        buf366 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf363, buf366, 12288, 128, grid=grid(12288), stream=stream0)
        buf367 = reinterpret_tensor(buf361, (1, 3072), (3072, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf366, buf367, 3072, 4, grid=grid(3072), stream=stream0)
        buf370 = reinterpret_tensor(buf348, (1, 512, 768), (393216, 768, 1), 0); del buf348  # reuse
        buf373 = reinterpret_tensor(buf344, (1, 512, 768), (393216, 768, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf355, buf364, primals_62, mul_25, div_52, getitem_37, buf370, buf373, 512, 768, grid=grid(512), stream=stream0)
        del div_52
        del getitem_37
        del primals_62
        buf371 = empty((768, ), device='cuda', dtype=torch.float32)
        buf372 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf355, buf364, mul_25, buf371, buf372, 768, 512, grid=grid(768), stream=stream0)
        del buf355
        del mul_25
        buf374 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (512, 768), (768, 1), 0), permute_414, out=buf374)
        del permute_414
        buf375 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (768, 512), (1, 768), 0), view_82, out=buf375)
        del view_82
        buf376 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf373, buf376, 3072, 128, grid=grid(3072), stream=stream0)
        buf377 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf376, buf377, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf378 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf374, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_183
        del getitem_184
        del getitem_185
        buf379 = buf378[0]
        buf380 = buf378[1]
        buf381 = buf378[2]
        del buf378
        buf382 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (512, 768), (768, 1), 0), permute_426, out=buf382)
        del permute_426
        buf383 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (768, 512), (1, 768), 0), view_66, out=buf383)
        buf384 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf381, buf384, 3072, 128, grid=grid(3072), stream=stream0)
        buf385 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf384, buf385, 768, 4, grid=grid(768), stream=stream0)
        buf386 = reinterpret_tensor(buf381, (512, 768), (768, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 768), (768, 1), 0), permute_431, out=buf386)
        del permute_431
        buf387 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (768, 512), (1, 768), 0), view_66, out=buf387)
        buf388 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf380, buf388, 3072, 128, grid=grid(3072), stream=stream0)
        buf389 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf388, buf389, 768, 4, grid=grid(768), stream=stream0)
        buf390 = reinterpret_tensor(buf380, (512, 768), (768, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (512, 768), (768, 1), 0), permute_435, out=buf390)
        del permute_435
        buf391 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (768, 512), (1, 768), 0), view_66, out=buf391)
        del view_66
        buf392 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf379, buf392, 3072, 128, grid=grid(3072), stream=stream0)
        buf393 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf392, buf393, 768, 4, grid=grid(768), stream=stream0)
        buf397 = reinterpret_tensor(buf379, (1, 512, 768), (393216, 768, 1), 0); del buf379  # reuse
        buf400 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf370, buf382, buf386, buf390, primals_52, mul_23, div_54, getitem_31, buf397, buf400, 512, 768, grid=grid(512), stream=stream0)
        del div_54
        del getitem_31
        del primals_52
        buf398 = empty((768, ), device='cuda', dtype=torch.float32)
        buf399 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf370, buf382, buf386, buf390, mul_23, buf398, buf399, 768, 512, grid=grid(768), stream=stream0)
        del buf370
        del buf382
        del mul_23
        buf401 = reinterpret_tensor(buf363, (512, 3072), (3072, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (512, 768), (768, 1), 0), permute_439, out=buf401)
        del permute_439
        buf402 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (768, 512), (1, 768), 0), view_64, out=buf402)
        del view_64
        buf403 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf400, buf403, 3072, 128, grid=grid(3072), stream=stream0)
        buf404 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf403, buf404, 768, 4, grid=grid(768), stream=stream0)
        buf405 = reinterpret_tensor(buf401, (1, 512, 3072), (1572864, 3072, 1), 0); del buf401  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf405, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf406 = reinterpret_tensor(buf400, (512, 768), (768, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 3072), (3072, 1), 0), permute_443, out=buf406)
        del permute_443
        buf407 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (3072, 512), (1, 3072), 0), view_62, out=buf407)
        del view_62
        buf408 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf405, buf408, 12288, 128, grid=grid(12288), stream=stream0)
        buf409 = reinterpret_tensor(buf403, (1, 3072), (3072, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf408, buf409, 3072, 4, grid=grid(3072), stream=stream0)
        buf412 = reinterpret_tensor(buf390, (1, 512, 768), (393216, 768, 1), 0); del buf390  # reuse
        buf415 = reinterpret_tensor(buf386, (1, 512, 768), (393216, 768, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf397, buf406, primals_46, mul_18, div_55, getitem_27, buf412, buf415, 512, 768, grid=grid(512), stream=stream0)
        del div_55
        del getitem_27
        del primals_46
        buf413 = empty((768, ), device='cuda', dtype=torch.float32)
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf397, buf406, mul_18, buf413, buf414, 768, 512, grid=grid(768), stream=stream0)
        del buf397
        del mul_18
        buf416 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 768), (768, 1), 0), permute_447, out=buf416)
        del permute_447
        buf417 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf415, (768, 512), (1, 768), 0), view_60, out=buf417)
        del view_60
        buf418 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf415, buf418, 3072, 128, grid=grid(3072), stream=stream0)
        buf419 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf418, buf419, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf420 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf416, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_190
        del getitem_191
        del getitem_192
        buf421 = buf420[0]
        buf422 = buf420[1]
        buf423 = buf420[2]
        del buf420
        buf424 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 768), (768, 1), 0), permute_459, out=buf424)
        del permute_459
        buf425 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (768, 512), (1, 768), 0), view_44, out=buf425)
        buf426 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf423, buf426, 3072, 128, grid=grid(3072), stream=stream0)
        buf427 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf426, buf427, 768, 4, grid=grid(768), stream=stream0)
        buf428 = reinterpret_tensor(buf423, (512, 768), (768, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (512, 768), (768, 1), 0), permute_464, out=buf428)
        del permute_464
        buf429 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_44, out=buf429)
        buf430 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf422, buf430, 3072, 128, grid=grid(3072), stream=stream0)
        buf431 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf430, buf431, 768, 4, grid=grid(768), stream=stream0)
        buf432 = reinterpret_tensor(buf422, (512, 768), (768, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (512, 768), (768, 1), 0), permute_468, out=buf432)
        del permute_468
        buf433 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (768, 512), (1, 768), 0), view_44, out=buf433)
        del view_44
        buf434 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf421, buf434, 3072, 128, grid=grid(3072), stream=stream0)
        buf435 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf434, buf435, 768, 4, grid=grid(768), stream=stream0)
        buf439 = reinterpret_tensor(buf421, (1, 512, 768), (393216, 768, 1), 0); del buf421  # reuse
        buf442 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf412, buf424, buf428, buf432, primals_36, mul_16, div_57, getitem_21, buf439, buf442, 512, 768, grid=grid(512), stream=stream0)
        del div_57
        del getitem_21
        del primals_36
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        buf441 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf412, buf424, buf428, buf432, mul_16, buf440, buf441, 768, 512, grid=grid(768), stream=stream0)
        del buf412
        del buf424
        del mul_16
        buf443 = reinterpret_tensor(buf405, (512, 3072), (3072, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (512, 768), (768, 1), 0), permute_472, out=buf443)
        del permute_472
        buf444 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (768, 512), (1, 768), 0), view_42, out=buf444)
        del view_42
        buf445 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf442, buf445, 3072, 128, grid=grid(3072), stream=stream0)
        buf446 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf445, buf446, 768, 4, grid=grid(768), stream=stream0)
        buf447 = reinterpret_tensor(buf443, (1, 512, 3072), (1572864, 3072, 1), 0); del buf443  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf447, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf448 = reinterpret_tensor(buf442, (512, 768), (768, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (512, 3072), (3072, 1), 0), permute_476, out=buf448)
        del permute_476
        buf449 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (3072, 512), (1, 3072), 0), view_40, out=buf449)
        del view_40
        buf450 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf447, buf450, 12288, 128, grid=grid(12288), stream=stream0)
        buf451 = reinterpret_tensor(buf445, (1, 3072), (3072, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf450, buf451, 3072, 4, grid=grid(3072), stream=stream0)
        buf454 = reinterpret_tensor(buf432, (1, 512, 768), (393216, 768, 1), 0); del buf432  # reuse
        buf457 = reinterpret_tensor(buf428, (1, 512, 768), (393216, 768, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf439, buf448, primals_30, mul_11, div_58, getitem_17, buf454, buf457, 512, 768, grid=grid(512), stream=stream0)
        del div_58
        del getitem_17
        del primals_30
        buf455 = empty((768, ), device='cuda', dtype=torch.float32)
        buf456 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf439, buf448, mul_11, buf455, buf456, 768, 512, grid=grid(768), stream=stream0)
        del buf439
        del mul_11
        buf458 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (512, 768), (768, 1), 0), permute_480, out=buf458)
        del permute_480
        buf459 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (768, 512), (1, 768), 0), view_38, out=buf459)
        del view_38
        buf460 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf457, buf460, 3072, 128, grid=grid(3072), stream=stream0)
        buf461 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf460, buf461, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf462 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf458, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_197
        del getitem_198
        del getitem_199
        buf463 = buf462[0]
        buf464 = buf462[1]
        buf465 = buf462[2]
        del buf462
        buf466 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 768), (768, 1), 0), permute_492, out=buf466)
        del permute_492
        buf467 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (768, 512), (1, 768), 0), view_22, out=buf467)
        buf468 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf465, buf468, 3072, 128, grid=grid(3072), stream=stream0)
        buf469 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf468, buf469, 768, 4, grid=grid(768), stream=stream0)
        buf470 = reinterpret_tensor(buf465, (512, 768), (768, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (512, 768), (768, 1), 0), permute_497, out=buf470)
        del permute_497
        buf471 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (768, 512), (1, 768), 0), view_22, out=buf471)
        buf472 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf464, buf472, 3072, 128, grid=grid(3072), stream=stream0)
        buf473 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf472, buf473, 768, 4, grid=grid(768), stream=stream0)
        buf474 = reinterpret_tensor(buf464, (512, 768), (768, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (512, 768), (768, 1), 0), permute_501, out=buf474)
        del permute_501
        buf475 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (768, 512), (1, 768), 0), view_22, out=buf475)
        del view_22
        buf476 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf463, buf476, 3072, 128, grid=grid(3072), stream=stream0)
        buf477 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf476, buf477, 768, 4, grid=grid(768), stream=stream0)
        buf481 = reinterpret_tensor(buf463, (1, 512, 768), (393216, 768, 1), 0); del buf463  # reuse
        buf484 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_14.run(buf454, buf466, buf470, buf474, primals_20, mul_9, div_60, getitem_11, buf481, buf484, 512, 768, grid=grid(512), stream=stream0)
        del div_60
        del getitem_11
        del primals_20
        buf482 = empty((768, ), device='cuda', dtype=torch.float32)
        buf483 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_15.run(buf454, buf466, buf470, buf474, mul_9, buf482, buf483, 768, 512, grid=grid(768), stream=stream0)
        del buf454
        del buf466
        del mul_9
        buf485 = reinterpret_tensor(buf447, (512, 3072), (3072, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (512, 768), (768, 1), 0), permute_505, out=buf485)
        del permute_505
        buf486 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (768, 512), (1, 768), 0), view_20, out=buf486)
        del view_20
        buf487 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf484, buf487, 3072, 128, grid=grid(3072), stream=stream0)
        buf488 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf487, buf488, 768, 4, grid=grid(768), stream=stream0)
        buf489 = reinterpret_tensor(buf485, (1, 512, 3072), (1572864, 3072, 1), 0); del buf485  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf489, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf490 = reinterpret_tensor(buf484, (512, 768), (768, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (512, 3072), (3072, 1), 0), permute_509, out=buf490)
        del permute_509
        buf491 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (3072, 512), (1, 3072), 0), view_18, out=buf491)
        del view_18
        buf492 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_10.run(buf489, buf492, 12288, 128, grid=grid(12288), stream=stream0)
        del buf489
        buf493 = reinterpret_tensor(buf487, (1, 3072), (3072, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_11.run(buf492, buf493, 3072, 4, grid=grid(3072), stream=stream0)
        del buf492
        buf496 = reinterpret_tensor(buf474, (1, 512, 768), (393216, 768, 1), 0); del buf474  # reuse
        buf499 = reinterpret_tensor(buf470, (1, 512, 768), (393216, 768, 1), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_12.run(buf481, buf490, primals_14, mul_4, div_61, getitem_7, buf496, buf499, 512, 768, grid=grid(512), stream=stream0)
        del div_61
        del getitem_7
        del primals_14
        buf497 = empty((768, ), device='cuda', dtype=torch.float32)
        buf498 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_13.run(buf481, buf490, mul_4, buf497, buf498, 768, 512, grid=grid(768), stream=stream0)
        del mul_4
        buf500 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (512, 768), (768, 1), 0), permute_513, out=buf500)
        del permute_513
        buf501 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf499, (768, 512), (1, 768), 0), view_16, out=buf501)
        del view_16
        buf502 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf499, buf502, 3072, 128, grid=grid(3072), stream=stream0)
        buf503 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf502, buf503, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf504 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf500, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_204
        del getitem_205
        del getitem_206
        buf505 = buf504[0]
        buf506 = buf504[1]
        buf507 = buf504[2]
        del buf504
        buf508 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 768), (768, 1), 0), permute_525, out=buf508)
        del permute_525
        buf509 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (768, 512), (1, 768), 0), view, out=buf509)
        buf510 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf507, buf510, 3072, 128, grid=grid(3072), stream=stream0)
        buf511 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf510, buf511, 768, 4, grid=grid(768), stream=stream0)
        buf512 = reinterpret_tensor(buf507, (512, 768), (768, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (512, 768), (768, 1), 0), permute_530, out=buf512)
        del permute_530
        buf513 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (768, 512), (1, 768), 0), view, out=buf513)
        buf514 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf506, buf514, 3072, 128, grid=grid(3072), stream=stream0)
        buf515 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf514, buf515, 768, 4, grid=grid(768), stream=stream0)
        buf516 = reinterpret_tensor(buf506, (512, 768), (768, 1), 0); del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (512, 768), (768, 1), 0), permute_534, out=buf516)
        del permute_534
        buf517 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (768, 512), (1, 768), 0), view, out=buf517)
        del view
        buf518 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf505, buf518, 3072, 128, grid=grid(3072), stream=stream0)
        buf519 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf518, buf519, 768, 4, grid=grid(768), stream=stream0)
        del buf518
        buf520 = buf496; del buf496  # reuse
        buf527 = reinterpret_tensor(buf505, (1, 512, 768), (393216, 768, 1), 0); del buf505  # reuse
        buf531 = buf499; del buf499  # reuse
        buf535 = buf481; del buf481  # reuse
        # Source Nodes: [loss], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_16.run(buf520, buf508, buf512, buf516, getitem_3, primals_4, mul_2, div_63, add_1, expand, primals_206, buf527, buf531, buf535, 512, 768, grid=grid(512), stream=stream0)
        del buf508
        del buf512
        del buf516
        del div_63
        del getitem_3
        del primals_4
        buf524 = empty((768, ), device='cuda', dtype=torch.float32)
        buf525 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf520, mul_2, buf524, buf525, 768, 512, grid=grid(768), stream=stream0)
        del mul_2
        buf526 = reinterpret_tensor(buf520, (512, 768), (768, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_17.run(buf526, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf526, [add_1], buf527, True)
        del add_1
        del buf527
        buf530 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf530, 1536, grid=grid(1536), stream=stream0)
        aten.index_put_(buf530, [expand], buf531, True)
        del buf531
        del expand
        buf534 = empty((50265, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_19.run(buf534, 38603520, grid=grid(38603520), stream=stream0)
        aten.index_put_(buf534, [primals_206], buf535, True)
        del buf535
        del primals_206
        return (buf534, buf530, buf526, buf524, buf525, reinterpret_tensor(buf517, (768, 768), (768, 1), 0), reinterpret_tensor(buf519, (768, ), (1, ), 0), reinterpret_tensor(buf513, (768, 768), (768, 1), 0), reinterpret_tensor(buf515, (768, ), (1, ), 0), reinterpret_tensor(buf509, (768, 768), (768, 1), 0), reinterpret_tensor(buf511, (768, ), (1, ), 0), reinterpret_tensor(buf501, (768, 768), (768, 1), 0), reinterpret_tensor(buf503, (768, ), (1, ), 0), buf497, buf498, reinterpret_tensor(buf491, (3072, 768), (768, 1), 0), reinterpret_tensor(buf493, (3072, ), (1, ), 0), reinterpret_tensor(buf486, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf488, (768, ), (1, ), 0), buf482, buf483, reinterpret_tensor(buf475, (768, 768), (768, 1), 0), reinterpret_tensor(buf477, (768, ), (1, ), 0), reinterpret_tensor(buf471, (768, 768), (768, 1), 0), reinterpret_tensor(buf473, (768, ), (1, ), 0), reinterpret_tensor(buf467, (768, 768), (768, 1), 0), reinterpret_tensor(buf469, (768, ), (1, ), 0), reinterpret_tensor(buf459, (768, 768), (768, 1), 0), reinterpret_tensor(buf461, (768, ), (1, ), 0), buf455, buf456, reinterpret_tensor(buf449, (3072, 768), (768, 1), 0), reinterpret_tensor(buf451, (3072, ), (1, ), 0), reinterpret_tensor(buf444, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf446, (768, ), (1, ), 0), buf440, buf441, reinterpret_tensor(buf433, (768, 768), (768, 1), 0), reinterpret_tensor(buf435, (768, ), (1, ), 0), reinterpret_tensor(buf429, (768, 768), (768, 1), 0), reinterpret_tensor(buf431, (768, ), (1, ), 0), reinterpret_tensor(buf425, (768, 768), (768, 1), 0), reinterpret_tensor(buf427, (768, ), (1, ), 0), reinterpret_tensor(buf417, (768, 768), (768, 1), 0), reinterpret_tensor(buf419, (768, ), (1, ), 0), buf413, buf414, reinterpret_tensor(buf407, (3072, 768), (768, 1), 0), reinterpret_tensor(buf409, (3072, ), (1, ), 0), reinterpret_tensor(buf402, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf404, (768, ), (1, ), 0), buf398, buf399, reinterpret_tensor(buf391, (768, 768), (768, 1), 0), reinterpret_tensor(buf393, (768, ), (1, ), 0), reinterpret_tensor(buf387, (768, 768), (768, 1), 0), reinterpret_tensor(buf389, (768, ), (1, ), 0), reinterpret_tensor(buf383, (768, 768), (768, 1), 0), reinterpret_tensor(buf385, (768, ), (1, ), 0), reinterpret_tensor(buf375, (768, 768), (768, 1), 0), reinterpret_tensor(buf377, (768, ), (1, ), 0), buf371, buf372, reinterpret_tensor(buf365, (3072, 768), (768, 1), 0), reinterpret_tensor(buf367, (3072, ), (1, ), 0), reinterpret_tensor(buf360, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf362, (768, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf349, (768, 768), (768, 1), 0), reinterpret_tensor(buf351, (768, ), (1, ), 0), reinterpret_tensor(buf345, (768, 768), (768, 1), 0), reinterpret_tensor(buf347, (768, ), (1, ), 0), reinterpret_tensor(buf341, (768, 768), (768, 1), 0), reinterpret_tensor(buf343, (768, ), (1, ), 0), reinterpret_tensor(buf333, (768, 768), (768, 1), 0), reinterpret_tensor(buf335, (768, ), (1, ), 0), buf329, buf330, reinterpret_tensor(buf323, (3072, 768), (768, 1), 0), reinterpret_tensor(buf325, (3072, ), (1, ), 0), reinterpret_tensor(buf318, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf320, (768, ), (1, ), 0), buf314, buf315, reinterpret_tensor(buf307, (768, 768), (768, 1), 0), reinterpret_tensor(buf309, (768, ), (1, ), 0), reinterpret_tensor(buf303, (768, 768), (768, 1), 0), reinterpret_tensor(buf305, (768, ), (1, ), 0), reinterpret_tensor(buf299, (768, 768), (768, 1), 0), reinterpret_tensor(buf301, (768, ), (1, ), 0), reinterpret_tensor(buf291, (768, 768), (768, 1), 0), reinterpret_tensor(buf293, (768, ), (1, ), 0), buf287, buf288, reinterpret_tensor(buf281, (3072, 768), (768, 1), 0), reinterpret_tensor(buf283, (3072, ), (1, ), 0), reinterpret_tensor(buf276, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf278, (768, ), (1, ), 0), buf272, buf273, reinterpret_tensor(buf265, (768, 768), (768, 1), 0), reinterpret_tensor(buf267, (768, ), (1, ), 0), reinterpret_tensor(buf261, (768, 768), (768, 1), 0), reinterpret_tensor(buf263, (768, ), (1, ), 0), reinterpret_tensor(buf257, (768, 768), (768, 1), 0), reinterpret_tensor(buf259, (768, ), (1, ), 0), reinterpret_tensor(buf249, (768, 768), (768, 1), 0), reinterpret_tensor(buf251, (768, ), (1, ), 0), buf245, buf246, reinterpret_tensor(buf239, (3072, 768), (768, 1), 0), reinterpret_tensor(buf241, (3072, ), (1, ), 0), reinterpret_tensor(buf234, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf236, (768, ), (1, ), 0), buf230, buf231, reinterpret_tensor(buf223, (768, 768), (768, 1), 0), reinterpret_tensor(buf225, (768, ), (1, ), 0), reinterpret_tensor(buf219, (768, 768), (768, 1), 0), reinterpret_tensor(buf221, (768, ), (1, ), 0), reinterpret_tensor(buf215, (768, 768), (768, 1), 0), reinterpret_tensor(buf217, (768, ), (1, ), 0), reinterpret_tensor(buf207, (768, 768), (768, 1), 0), reinterpret_tensor(buf209, (768, ), (1, ), 0), buf203, buf204, reinterpret_tensor(buf197, (3072, 768), (768, 1), 0), reinterpret_tensor(buf199, (3072, ), (1, ), 0), reinterpret_tensor(buf192, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf194, (768, ), (1, ), 0), buf188, buf189, reinterpret_tensor(buf181, (768, 768), (768, 1), 0), reinterpret_tensor(buf183, (768, ), (1, ), 0), reinterpret_tensor(buf177, (768, 768), (768, 1), 0), reinterpret_tensor(buf179, (768, ), (1, ), 0), reinterpret_tensor(buf173, (768, 768), (768, 1), 0), reinterpret_tensor(buf175, (768, ), (1, ), 0), reinterpret_tensor(buf165, (768, 768), (768, 1), 0), reinterpret_tensor(buf167, (768, ), (1, ), 0), buf161, buf162, reinterpret_tensor(buf155, (3072, 768), (768, 1), 0), reinterpret_tensor(buf157, (3072, ), (1, ), 0), reinterpret_tensor(buf150, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf152, (768, ), (1, ), 0), buf146, buf147, reinterpret_tensor(buf139, (768, 768), (768, 1), 0), reinterpret_tensor(buf141, (768, ), (1, ), 0), reinterpret_tensor(buf135, (768, 768), (768, 1), 0), reinterpret_tensor(buf137, (768, ), (1, ), 0), reinterpret_tensor(buf131, (768, 768), (768, 1), 0), reinterpret_tensor(buf133, (768, ), (1, ), 0), reinterpret_tensor(buf123, (768, 768), (768, 1), 0), reinterpret_tensor(buf125, (768, ), (1, ), 0), buf119, buf120, reinterpret_tensor(buf113, (3072, 768), (768, 1), 0), reinterpret_tensor(buf115, (3072, ), (1, ), 0), reinterpret_tensor(buf108, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), buf104, buf105, reinterpret_tensor(buf97, (768, 768), (768, 1), 0), reinterpret_tensor(buf99, (768, ), (1, ), 0), reinterpret_tensor(buf93, (768, 768), (768, 1), 0), reinterpret_tensor(buf95, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, 768), (768, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), reinterpret_tensor(buf81, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), buf77, buf78, reinterpret_tensor(buf71, (3072, 768), (768, 1), 0), reinterpret_tensor(buf73, (3072, ), (1, ), 0), reinterpret_tensor(buf66, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf68, (768, ), (1, ), 0), buf62, buf63, reinterpret_tensor(buf55, (768, 768), (768, 1), 0), reinterpret_tensor(buf57, (768, ), (1, ), 0), reinterpret_tensor(buf51, (768, 768), (768, 1), 0), reinterpret_tensor(buf53, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, 768), (768, 1), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf39, (768, 768), (768, 1), 0), reinterpret_tensor(buf41, (768, ), (1, ), 0), buf35, buf36, reinterpret_tensor(buf29, (3072, 768), (768, 1), 0), reinterpret_tensor(buf31, (3072, ), (1, ), 0), reinterpret_tensor(buf24, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), buf20, buf21, reinterpret_tensor(buf14, (768, 768), (768, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf6, (50265, 768), (768, 1), 0), reinterpret_tensor(buf7, (50265, ), (1, ), 0), None, None, None, )


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
    primals_206 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
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
    sub_40 = rand_strided((511, 50265), (50265, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_4 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_2 = rand_strided((511, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_134 = rand_strided((50265, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
    tangents_2 = rand_strided((1, 512, 50265), (25735680, 50265, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, add_1, mul_2, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_4, view_18, addmm_4, view_20, getitem_11, mul_9, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_11, view_40, addmm_10, view_42, getitem_21, mul_16, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_18, view_62, addmm_16, view_64, getitem_31, mul_23, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_25, view_84, addmm_22, view_86, getitem_41, mul_30, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_32, view_106, addmm_28, view_108, getitem_51, mul_37, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_39, view_128, addmm_34, view_130, getitem_61, mul_44, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_46, view_150, addmm_40, view_152, getitem_71, mul_51, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_53, view_172, addmm_46, view_174, getitem_81, mul_58, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_60, view_194, addmm_52, view_196, getitem_91, mul_65, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_67, view_216, addmm_58, view_218, getitem_101, mul_72, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_74, view_238, addmm_64, view_240, getitem_111, mul_79, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_81, view_260, addmm_70, view_262, getitem_121, mul_86, view_264, addmm_72, mul_91, view_266, sub_40, convert_element_type_3, ne_4, where_2, permute_134, div_26, permute_138, div_27, permute_142, permute_146, div_28, permute_150, permute_162, permute_167, permute_171, div_30, permute_175, permute_179, div_31, permute_183, permute_195, permute_200, permute_204, div_33, permute_208, permute_212, div_34, permute_216, permute_228, permute_233, permute_237, div_36, permute_241, permute_245, div_37, permute_249, permute_261, permute_266, permute_270, div_39, permute_274, permute_278, div_40, permute_282, permute_294, permute_299, permute_303, div_42, permute_307, permute_311, div_43, permute_315, permute_327, permute_332, permute_336, div_45, permute_340, permute_344, div_46, permute_348, permute_360, permute_365, permute_369, div_48, permute_373, permute_377, div_49, permute_381, permute_393, permute_398, permute_402, div_51, permute_406, permute_410, div_52, permute_414, permute_426, permute_431, permute_435, div_54, permute_439, permute_443, div_55, permute_447, permute_459, permute_464, permute_468, div_57, permute_472, permute_476, div_58, permute_480, permute_492, permute_497, permute_501, div_60, permute_505, permute_509, div_61, permute_513, permute_525, permute_530, permute_534, div_63, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('RobertaForCausalLM', benchmark_compiled_module)
