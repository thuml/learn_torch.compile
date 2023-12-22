
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


# kernel path: /tmp/torchinductor_youkaichao/43/c43mstufmlfco6eg4pmy2uvuyrfyrwcsqvfhtq7uaviy5pwxnyew.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 30522
    rnumel = 2048
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqgxbjrwg5kutzujlujxw5d2nwhlcgehg37dftypm54epsypatt.py
# Source Nodes: [hidden_states_109], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
# hidden_states_109 => add_100, erf_12, mul_88
triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_1', 'mutated_arg_names': []}
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
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
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22ygqsc6qdh6kmwvlt4bsv3bo3oozcejne2evsxuaba5flsnqqi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jj/cjju6yq7ulvkxe5wfwigfs5v2bke6cukucxgn2k332aivatg4yr7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sp/cspilsqdoxlkkg67saqdz3zxx2o4cr6scuuv4p5hhkofsligd7xp.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vi/cvih22dh4pfnvghcwvh4wlfzbozh5c3cn6ptbonm5uirshpx3vub.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfkijezetz5uhishzoiqsgditm4nndoq425vvqwxe6ljybiwfvr.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_96, erf_11, mul_83
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/3i/c3i2ncokrnxplqmug4lhiabn7m67zvf2ihdwvpfduv73lym6jzb4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ka/cka6vdro6hjtg5eys5mykajcjv2i7ekph4qmrgf6pgz5yl45e2fl.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3mq3kl7fhoublsahxp6avf6irygkf4a7xhutii63qze3aavmb4.py
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
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgs3q36fvaplwx7hd4sgfttwgchs67c5y77yzhre6cj5ycjlnarv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2r/c2r74hnklqqlji6mxdc6u2p6z2fc5rly4zlvhmfrhrkzvlefiilk.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6k/c6km5zfakjdlyx5gidzbgk3xuclnkozfmrmytupq2pgjm6gjtpts.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czoxla6xul5y6ak57u25scnmq3nndribzdmuvk2wdf6npiquq3wp.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]

triton_per_fused__softmax_backward_data_div_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_13', 'mutated_arg_names': []}
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
    tmp9 = 8.0
    tmp10 = tmp8 / tmp9
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp10, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32najj3xnedo7radydqd2cosijt2hfgbt5vzlo2yxzyhxo5rqmi.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yh/cyhmpswqhriurocds2p6tmgdgp3rg4jp7y2t6dmhppccc3bvxvoh.py
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhi4voq3xivena3tcmfjq2wcyr2cmhfxi6jf6y7zrbq53fbke7r.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_layer_norm_backward]

triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x2 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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
    tmp27 = tl.full([1], -1, tl.int64)
    tmp28 = tmp26 == tmp27
    tmp29 = 0.0
    tmp30 = tl.where(tmp28, tmp29, tmp25)
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 == tmp32
    tmp34 = tl.where(tmp33, tmp29, tmp25)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp25, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp30, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp34, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7nk76eut3mbsvvhbbh72gkviajmu7noq7pa4e3wbjn7ujlltb2.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rc/crcibofpir7u6yv4qxk3klmcai2t7hezpbz5jc3tycpg5nrzqwon.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]

triton_poi_fused_embedding_dense_backward_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_sum_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (393216 + x2), None)
    tmp6 = tl.load(in_ptr1 + (786432 + x2), None)
    tmp8 = tl.load(in_ptr1 + (1179648 + x2), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tl.store(out_ptr0 + (x2), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbb7f4rzwkxah37732dnyp5xci44ivuhvb7qpeugtbxpsejc3gri.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_21', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfkhtqtgu5gixkdm2r3nx6nqc32esrnwpoz5idfgsfddzgh4jrp.py
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
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_22', 'mutated_arg_names': []},
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, slice_4, mul_1, view, view_16, mul_3, view_18, addmm_4, view_20, mul_8, view_22, view_38, mul_10, view_40, addmm_10, view_42, mul_15, view_44, view_60, mul_17, view_62, addmm_16, view_64, mul_22, view_66, view_82, mul_24, view_84, addmm_22, view_86, mul_29, view_88, view_104, mul_31, view_106, addmm_28, view_108, mul_36, view_110, view_126, mul_38, view_128, addmm_34, view_130, mul_43, view_132, view_148, mul_45, view_150, addmm_40, view_152, mul_50, view_154, view_170, mul_52, view_172, addmm_46, view_174, mul_57, view_176, view_192, mul_59, view_194, addmm_52, view_196, mul_64, view_198, view_214, mul_66, view_216, addmm_58, view_218, mul_71, view_220, view_236, mul_73, view_238, addmm_64, view_240, mul_78, view_242, view_258, mul_80, view_260, addmm_70, view_262, mul_85, view_264, addmm_72, mul_90, view_266, permute_134, div_24, permute_138, div_25, permute_142, permute_146, div_26, permute_150, permute_155, permute_156, alias_12, permute_157, permute_158, permute_162, permute_167, permute_171, div_28, permute_175, permute_179, div_29, permute_183, permute_188, permute_189, alias_13, permute_190, permute_191, permute_195, permute_200, permute_204, div_31, permute_208, permute_212, div_32, permute_216, permute_221, permute_222, alias_14, permute_223, permute_224, permute_228, permute_233, permute_237, div_34, permute_241, permute_245, div_35, permute_249, permute_254, permute_255, alias_15, permute_256, permute_257, permute_261, permute_266, permute_270, div_37, permute_274, permute_278, div_38, permute_282, permute_287, permute_288, alias_16, permute_289, permute_290, permute_294, permute_299, permute_303, div_40, permute_307, permute_311, div_41, permute_315, permute_320, permute_321, alias_17, permute_322, permute_323, permute_327, permute_332, permute_336, div_43, permute_340, permute_344, div_44, permute_348, permute_353, permute_354, alias_18, permute_355, permute_356, permute_360, permute_365, permute_369, div_46, permute_373, permute_377, div_47, permute_381, permute_386, permute_387, alias_19, permute_388, permute_389, permute_393, permute_398, permute_402, div_49, permute_406, permute_410, div_50, permute_414, permute_419, permute_420, alias_20, permute_421, permute_422, permute_426, permute_431, permute_435, div_52, permute_439, permute_443, div_53, permute_447, permute_452, permute_453, alias_21, permute_454, permute_455, permute_459, permute_464, permute_468, div_55, permute_472, permute_476, div_56, permute_480, permute_485, permute_486, alias_22, permute_487, permute_488, permute_492, permute_497, permute_501, div_58, permute_505, permute_509, div_59, permute_513, permute_518, permute_519, alias_23, permute_520, permute_521, permute_525, permute_530, permute_534, div_61, tangents_1 = args
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
    assert_size_stride(primals_206, (4, 512), (512, 1))
    assert_size_stride(expand, (4, 512), (0, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (2048, 768), (768, 1))
    assert_size_stride(view_16, (2048, 768), (768, 1))
    assert_size_stride(mul_3, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (2048, 768), (768, 1))
    assert_size_stride(addmm_4, (2048, 3072), (3072, 1))
    assert_size_stride(view_20, (2048, 3072), (3072, 1))
    assert_size_stride(mul_8, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (2048, 768), (768, 1))
    assert_size_stride(view_38, (2048, 768), (768, 1))
    assert_size_stride(mul_10, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_40, (2048, 768), (768, 1))
    assert_size_stride(addmm_10, (2048, 3072), (3072, 1))
    assert_size_stride(view_42, (2048, 3072), (3072, 1))
    assert_size_stride(mul_15, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_44, (2048, 768), (768, 1))
    assert_size_stride(view_60, (2048, 768), (768, 1))
    assert_size_stride(mul_17, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_62, (2048, 768), (768, 1))
    assert_size_stride(addmm_16, (2048, 3072), (3072, 1))
    assert_size_stride(view_64, (2048, 3072), (3072, 1))
    assert_size_stride(mul_22, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_66, (2048, 768), (768, 1))
    assert_size_stride(view_82, (2048, 768), (768, 1))
    assert_size_stride(mul_24, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_84, (2048, 768), (768, 1))
    assert_size_stride(addmm_22, (2048, 3072), (3072, 1))
    assert_size_stride(view_86, (2048, 3072), (3072, 1))
    assert_size_stride(mul_29, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_88, (2048, 768), (768, 1))
    assert_size_stride(view_104, (2048, 768), (768, 1))
    assert_size_stride(mul_31, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_106, (2048, 768), (768, 1))
    assert_size_stride(addmm_28, (2048, 3072), (3072, 1))
    assert_size_stride(view_108, (2048, 3072), (3072, 1))
    assert_size_stride(mul_36, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_110, (2048, 768), (768, 1))
    assert_size_stride(view_126, (2048, 768), (768, 1))
    assert_size_stride(mul_38, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_128, (2048, 768), (768, 1))
    assert_size_stride(addmm_34, (2048, 3072), (3072, 1))
    assert_size_stride(view_130, (2048, 3072), (3072, 1))
    assert_size_stride(mul_43, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_132, (2048, 768), (768, 1))
    assert_size_stride(view_148, (2048, 768), (768, 1))
    assert_size_stride(mul_45, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_150, (2048, 768), (768, 1))
    assert_size_stride(addmm_40, (2048, 3072), (3072, 1))
    assert_size_stride(view_152, (2048, 3072), (3072, 1))
    assert_size_stride(mul_50, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_154, (2048, 768), (768, 1))
    assert_size_stride(view_170, (2048, 768), (768, 1))
    assert_size_stride(mul_52, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_172, (2048, 768), (768, 1))
    assert_size_stride(addmm_46, (2048, 3072), (3072, 1))
    assert_size_stride(view_174, (2048, 3072), (3072, 1))
    assert_size_stride(mul_57, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (2048, 768), (768, 1))
    assert_size_stride(view_192, (2048, 768), (768, 1))
    assert_size_stride(mul_59, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_194, (2048, 768), (768, 1))
    assert_size_stride(addmm_52, (2048, 3072), (3072, 1))
    assert_size_stride(view_196, (2048, 3072), (3072, 1))
    assert_size_stride(mul_64, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_198, (2048, 768), (768, 1))
    assert_size_stride(view_214, (2048, 768), (768, 1))
    assert_size_stride(mul_66, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (2048, 768), (768, 1))
    assert_size_stride(addmm_58, (2048, 3072), (3072, 1))
    assert_size_stride(view_218, (2048, 3072), (3072, 1))
    assert_size_stride(mul_71, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_220, (2048, 768), (768, 1))
    assert_size_stride(view_236, (2048, 768), (768, 1))
    assert_size_stride(mul_73, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_238, (2048, 768), (768, 1))
    assert_size_stride(addmm_64, (2048, 3072), (3072, 1))
    assert_size_stride(view_240, (2048, 3072), (3072, 1))
    assert_size_stride(mul_78, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_242, (2048, 768), (768, 1))
    assert_size_stride(view_258, (2048, 768), (768, 1))
    assert_size_stride(mul_80, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_260, (2048, 768), (768, 1))
    assert_size_stride(addmm_70, (2048, 3072), (3072, 1))
    assert_size_stride(view_262, (2048, 3072), (3072, 1))
    assert_size_stride(mul_85, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_264, (2048, 768), (768, 1))
    assert_size_stride(addmm_72, (2048, 768), (768, 1))
    assert_size_stride(mul_90, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_266, (2048, 768), (768, 1))
    assert_size_stride(permute_134, (30522, 768), (768, 1))
    assert_size_stride(div_24, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_26, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_155, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_156, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_12, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_157, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_158, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_167, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_28, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_29, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_188, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_189, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_13, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_190, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_191, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_200, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_31, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_32, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_221, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_222, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_14, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_223, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_224, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_34, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_35, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_254, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_255, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_15, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_256, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_257, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_37, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_38, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_287, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_288, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_16, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_289, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_290, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_40, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_41, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_320, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_321, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_17, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_322, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_323, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_43, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 3072), (3072, 1))
    assert_size_stride(permute_344, (3072, 768), (768, 1))
    assert_size_stride(div_44, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (768, 768), (768, 1))
    assert_size_stride(permute_353, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_354, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_18, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_355, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_356, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_365, (768, 768), (768, 1))
    assert_size_stride(permute_369, (768, 768), (768, 1))
    assert_size_stride(div_46, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 3072), (3072, 1))
    assert_size_stride(permute_377, (3072, 768), (768, 1))
    assert_size_stride(div_47, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (768, 768), (768, 1))
    assert_size_stride(permute_386, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_387, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_19, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_388, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_389, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_402, (768, 768), (768, 1))
    assert_size_stride(div_49, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (768, 3072), (3072, 1))
    assert_size_stride(permute_410, (3072, 768), (768, 1))
    assert_size_stride(div_50, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (768, 768), (768, 1))
    assert_size_stride(permute_419, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_420, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_20, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_421, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_422, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_435, (768, 768), (768, 1))
    assert_size_stride(div_52, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (768, 3072), (3072, 1))
    assert_size_stride(permute_443, (3072, 768), (768, 1))
    assert_size_stride(div_53, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (768, 768), (768, 1))
    assert_size_stride(permute_452, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_453, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_21, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_454, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_455, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_464, (768, 768), (768, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(div_55, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (768, 3072), (3072, 1))
    assert_size_stride(permute_476, (3072, 768), (768, 1))
    assert_size_stride(div_56, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (768, 768), (768, 1))
    assert_size_stride(permute_485, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_486, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_22, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_487, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_488, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_497, (768, 768), (768, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(div_58, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (768, 3072), (3072, 1))
    assert_size_stride(permute_509, (3072, 768), (768, 1))
    assert_size_stride(div_59, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (768, 768), (768, 1))
    assert_size_stride(permute_518, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_519, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_23, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_520, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_521, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_530, (768, 768), (768, 1))
    assert_size_stride(permute_534, (768, 768), (768, 1))
    assert_size_stride(div_61, (4, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (4, 512, 30522), (15627264, 30522, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (2048, 30522), (30522, 1), 0), permute_134, out=buf0)
        del permute_134
        buf1 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (30522, 2048), (1, 30522), 0), view_266, out=buf1)
        del view_266
        buf2 = empty((1, 30522), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 30522, 2048, grid=grid(30522), stream=stream0)
        del tangents_1
        buf9 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [hidden_states_109], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_backward_1.run(buf0, primals_200, mul_90, div_24, addmm_72, buf9, 2048, 768, grid=grid(2048), stream=stream0)
        del addmm_72
        del div_24
        del primals_200
        buf5 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_2.run(buf0, mul_90, buf5, buf7, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_90
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf5, buf6, 768, 16, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf7, buf8, 768, 16, grid=grid(768), stream=stream0)
        buf10 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (2048, 768), (768, 1), 0), permute_138, out=buf10)
        del permute_138
        buf11 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (768, 2048), (1, 768), 0), view_264, out=buf11)
        del view_264
        buf12 = reinterpret_tensor(buf7, (1, 768, 16), (12288, 1, 768), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf9, buf12, 12288, 128, grid=grid(12288), stream=stream0)
        buf13 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf12, buf13, 768, 16, grid=grid(768), stream=stream0)
        buf16 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf10, primals_196, mul_85, div_25, buf16, 2048, 768, grid=grid(2048), stream=stream0)
        del div_25
        del primals_196
        buf17 = reinterpret_tensor(buf12, (768, 16), (1, 768), 0); del buf12  # reuse
        buf19 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_2.run(buf10, mul_85, buf17, buf19, 12288, 128, grid=grid(12288), stream=stream0)
        del mul_85
        buf18 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf17, buf18, 768, 16, grid=grid(768), stream=stream0)
        buf20 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf19, buf20, 768, 16, grid=grid(768), stream=stream0)
        buf21 = empty((2048, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (2048, 768), (768, 1), 0), permute_142, out=buf21)
        del permute_142
        buf22 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (768, 2048), (1, 768), 0), view_262, out=buf22)
        del view_262
        buf25 = reinterpret_tensor(buf21, (4, 512, 3072), (1572864, 3072, 1), 0); del buf21  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf25, addmm_70, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_70
        buf26 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (2048, 3072), (3072, 1), 0), permute_146, out=buf26)
        del permute_146
        buf23 = reinterpret_tensor(buf19, (1, 768, 16), (12288, 1, 768), 0); del buf19  # reuse
        buf33 = buf17; del buf17  # reuse
        buf35 = empty_strided((768, 16), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf16, buf26, mul_80, buf23, buf33, buf35, 12288, 128, grid=grid(12288), stream=stream0)
        buf24 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf23, buf24, 768, 16, grid=grid(768), stream=stream0)
        buf27 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (3072, 2048), (1, 3072), 0), view_260, out=buf27)
        del view_260
        buf28 = empty_strided((1, 3072, 16), (49152, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf25, buf28, 49152, 128, grid=grid(49152), stream=stream0)
        buf29 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf28, buf29, 3072, 16, grid=grid(3072), stream=stream0)
        buf32 = empty((4, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf16, buf26, primals_190, mul_80, div_26, buf32, 2048, 768, grid=grid(2048), stream=stream0)
        del div_26
        del mul_80
        del primals_190
        buf34 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf33, buf34, 768, 16, grid=grid(768), stream=stream0)
        buf36 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf35, buf36, 768, 16, grid=grid(768), stream=stream0)
        buf37 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (2048, 768), (768, 1), 0), permute_150, out=buf37)
        del permute_150
        buf38 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (768, 2048), (1, 768), 0), view_258, out=buf38)
        del view_258
        buf41 = reinterpret_tensor(buf16, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf37, buf41, 1572864, grid=grid(1572864), stream=stream0)
        buf42 = reinterpret_tensor(buf37, (48, 512, 64), (32768, 64, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_155, reinterpret_tensor(buf41, (48, 512, 64), (32768, 64, 1), 0), out=buf42)
        del permute_155
        buf48 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf42, buf48, 1572864, grid=grid(1572864), stream=stream0)
        buf49 = reinterpret_tensor(buf42, (2048, 768), (768, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf48, permute_162, out=buf49)
        del permute_162
        buf43 = empty((48, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (48, 512, 64), (32768, 64, 1), 0), permute_156, out=buf43)
        del permute_156
        buf45 = empty((4, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf43, alias_12, buf45, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_12
        buf46 = reinterpret_tensor(buf41, (48, 64, 512), (32768, 512, 1), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_157, reinterpret_tensor(buf45, (48, 512, 512), (262144, 512, 1), 0), out=buf46)
        del permute_157
        buf53 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf46, buf53, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf54 = reinterpret_tensor(buf46, (2048, 768), (768, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf53, permute_167, out=buf54)
        del permute_167
        buf47 = empty((48, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (48, 512, 512), (262144, 512, 1), 0), permute_158, out=buf47)
        del permute_158
        buf58 = empty((2048, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf47, buf58, 1572864, grid=grid(1572864), stream=stream0)
        buf59 = reinterpret_tensor(buf47, (2048, 768), (768, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf58, permute_171, out=buf59)
        del permute_171
        buf39 = reinterpret_tensor(buf35, (1, 768, 16), (12288, 1, 768), 0); del buf35  # reuse
        buf67 = buf33; del buf33  # reuse
        buf69 = reinterpret_tensor(buf23, (768, 16), (1, 768), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf32, buf49, buf54, buf59, mul_78, buf39, buf67, buf69, 12288, 128, grid=grid(12288), stream=stream0)
        buf40 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf39, buf40, 768, 16, grid=grid(768), stream=stream0)
        buf50 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (768, 2048), (1, 768), 0), view_242, out=buf50)
        buf51 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf48, buf51, 12288, 128, grid=grid(12288), stream=stream0)
        buf52 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf51, buf52, 768, 16, grid=grid(768), stream=stream0)
        buf55 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (768, 2048), (1, 768), 0), view_242, out=buf55)
        buf56 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf53, buf56, 12288, 128, grid=grid(12288), stream=stream0)
        buf57 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf56, buf57, 768, 16, grid=grid(768), stream=stream0)
        buf60 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (768, 2048), (1, 768), 0), view_242, out=buf60)
        del view_242
        buf61 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf58, buf61, 12288, 128, grid=grid(12288), stream=stream0)
        buf62 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf61, buf62, 768, 16, grid=grid(768), stream=stream0)
        buf63 = buf32; del buf32  # reuse
        buf66 = reinterpret_tensor(buf58, (4, 512, 768), (393216, 768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf63, buf49, buf54, buf59, primals_180, mul_78, div_28, buf66, 2048, 768, grid=grid(2048), stream=stream0)
        del div_28
        del mul_78
        del primals_180
        buf68 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf67, buf68, 768, 16, grid=grid(768), stream=stream0)
        buf70 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf69, buf70, 768, 16, grid=grid(768), stream=stream0)
        buf71 = reinterpret_tensor(buf25, (2048, 3072), (3072, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (2048, 768), (768, 1), 0), permute_175, out=buf71)
        del permute_175
        buf72 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (768, 2048), (1, 768), 0), view_240, out=buf72)
        del view_240
        buf75 = reinterpret_tensor(buf71, (4, 512, 3072), (1572864, 3072, 1), 0); del buf71  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf75, addmm_64, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_64
        buf76 = reinterpret_tensor(buf63, (2048, 768), (768, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (2048, 3072), (3072, 1), 0), permute_179, out=buf76)
        del permute_179
        buf73 = reinterpret_tensor(buf69, (1, 768, 16), (12288, 1, 768), 0); del buf69  # reuse
        buf83 = buf67; del buf67  # reuse
        buf85 = reinterpret_tensor(buf61, (768, 16), (1, 768), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf66, buf76, mul_73, buf73, buf83, buf85, 12288, 128, grid=grid(12288), stream=stream0)
        buf74 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf73, buf74, 768, 16, grid=grid(768), stream=stream0)
        buf77 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (3072, 2048), (1, 3072), 0), view_238, out=buf77)
        del view_238
        buf78 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf75, buf78, 49152, 128, grid=grid(49152), stream=stream0)
        buf79 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf78, buf79, 3072, 16, grid=grid(3072), stream=stream0)
        buf82 = reinterpret_tensor(buf59, (4, 512, 768), (393216, 768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf66, buf76, primals_174, mul_73, div_29, buf82, 2048, 768, grid=grid(2048), stream=stream0)
        del div_29
        del mul_73
        del primals_174
        buf84 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf83, buf84, 768, 16, grid=grid(768), stream=stream0)
        buf86 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf85, buf86, 768, 16, grid=grid(768), stream=stream0)
        buf87 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (2048, 768), (768, 1), 0), permute_183, out=buf87)
        del permute_183
        buf88 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (768, 2048), (1, 768), 0), view_236, out=buf88)
        del view_236
        buf91 = reinterpret_tensor(buf66, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf87, buf91, 1572864, grid=grid(1572864), stream=stream0)
        buf93 = reinterpret_tensor(buf45, (48, 512, 512), (262144, 512, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (48, 512, 64), (32768, 64, 1), 0), permute_189, out=buf93)
        del permute_189
        buf95 = reinterpret_tensor(buf43, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf93, alias_13, buf95, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_13
        buf96 = reinterpret_tensor(buf87, (48, 64, 512), (32768, 512, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_190, reinterpret_tensor(buf95, (48, 512, 512), (262144, 512, 1), 0), out=buf96)
        del permute_190
        buf103 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf96, buf103, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf104 = reinterpret_tensor(buf96, (2048, 768), (768, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf103, permute_200, out=buf104)
        del permute_200
        buf97 = reinterpret_tensor(buf49, (48, 512, 64), (32768, 64, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (48, 512, 512), (262144, 512, 1), 0), permute_191, out=buf97)
        del permute_191
        buf108 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf97, buf108, 1572864, grid=grid(1572864), stream=stream0)
        buf109 = reinterpret_tensor(buf97, (2048, 768), (768, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf108, permute_204, out=buf109)
        del permute_204
        buf92 = reinterpret_tensor(buf48, (48, 512, 64), (32768, 64, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_188, reinterpret_tensor(buf91, (48, 512, 64), (32768, 64, 1), 0), out=buf92)
        del permute_188
        buf98 = reinterpret_tensor(buf91, (2048, 768), (768, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf92, buf98, 1572864, grid=grid(1572864), stream=stream0)
        buf99 = reinterpret_tensor(buf92, (2048, 768), (768, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf98, permute_195, out=buf99)
        del permute_195
        buf89 = reinterpret_tensor(buf85, (1, 768, 16), (12288, 1, 768), 0); del buf85  # reuse
        buf117 = buf83; del buf83  # reuse
        buf119 = reinterpret_tensor(buf73, (768, 16), (1, 768), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf82, buf99, buf104, buf109, mul_71, buf89, buf117, buf119, 12288, 128, grid=grid(12288), stream=stream0)
        buf90 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf89, buf90, 768, 16, grid=grid(768), stream=stream0)
        buf100 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (768, 2048), (1, 768), 0), view_220, out=buf100)
        buf101 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf98, buf101, 12288, 128, grid=grid(12288), stream=stream0)
        buf102 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf101, buf102, 768, 16, grid=grid(768), stream=stream0)
        buf105 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (768, 2048), (1, 768), 0), view_220, out=buf105)
        buf106 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf103, buf106, 12288, 128, grid=grid(12288), stream=stream0)
        buf107 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf106, buf107, 768, 16, grid=grid(768), stream=stream0)
        buf110 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (768, 2048), (1, 768), 0), view_220, out=buf110)
        del view_220
        buf111 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf108, buf111, 12288, 128, grid=grid(12288), stream=stream0)
        buf112 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf111, buf112, 768, 16, grid=grid(768), stream=stream0)
        buf113 = reinterpret_tensor(buf104, (4, 512, 768), (393216, 768, 1), 0); del buf104  # reuse
        buf116 = reinterpret_tensor(buf108, (4, 512, 768), (393216, 768, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_17.run(buf113, buf82, buf99, buf109, primals_164, mul_71, div_31, buf116, 2048, 768, grid=grid(2048), stream=stream0)
        del div_31
        del mul_71
        del primals_164
        buf118 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf117, buf118, 768, 16, grid=grid(768), stream=stream0)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf119, buf120, 768, 16, grid=grid(768), stream=stream0)
        buf121 = reinterpret_tensor(buf75, (2048, 3072), (3072, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (2048, 768), (768, 1), 0), permute_208, out=buf121)
        del permute_208
        buf122 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (768, 2048), (1, 768), 0), view_218, out=buf122)
        del view_218
        buf125 = reinterpret_tensor(buf121, (4, 512, 3072), (1572864, 3072, 1), 0); del buf121  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf125, addmm_58, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_58
        buf126 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (2048, 3072), (3072, 1), 0), permute_212, out=buf126)
        del permute_212
        buf123 = reinterpret_tensor(buf119, (1, 768, 16), (12288, 1, 768), 0); del buf119  # reuse
        buf133 = buf117; del buf117  # reuse
        buf135 = reinterpret_tensor(buf111, (768, 16), (1, 768), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf116, buf126, mul_66, buf123, buf133, buf135, 12288, 128, grid=grid(12288), stream=stream0)
        buf124 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf123, buf124, 768, 16, grid=grid(768), stream=stream0)
        buf127 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (3072, 2048), (1, 3072), 0), view_216, out=buf127)
        del view_216
        buf128 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf125, buf128, 49152, 128, grid=grid(49152), stream=stream0)
        buf129 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf128, buf129, 3072, 16, grid=grid(3072), stream=stream0)
        buf132 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf116, buf126, primals_158, mul_66, div_32, buf132, 2048, 768, grid=grid(2048), stream=stream0)
        del div_32
        del mul_66
        del primals_158
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf133, buf134, 768, 16, grid=grid(768), stream=stream0)
        buf136 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf135, buf136, 768, 16, grid=grid(768), stream=stream0)
        buf137 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (2048, 768), (768, 1), 0), permute_216, out=buf137)
        del permute_216
        buf138 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (768, 2048), (1, 768), 0), view_214, out=buf138)
        del view_214
        buf141 = reinterpret_tensor(buf116, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf137, buf141, 1572864, grid=grid(1572864), stream=stream0)
        buf142 = reinterpret_tensor(buf137, (48, 512, 64), (32768, 64, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_221, reinterpret_tensor(buf141, (48, 512, 64), (32768, 64, 1), 0), out=buf142)
        del permute_221
        buf148 = reinterpret_tensor(buf113, (2048, 768), (768, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf142, buf148, 1572864, grid=grid(1572864), stream=stream0)
        buf149 = reinterpret_tensor(buf142, (2048, 768), (768, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf148, permute_228, out=buf149)
        del permute_228
        buf143 = reinterpret_tensor(buf95, (48, 512, 512), (262144, 512, 1), 0); del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (48, 512, 64), (32768, 64, 1), 0), permute_222, out=buf143)
        del permute_222
        buf145 = reinterpret_tensor(buf93, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf143, alias_14, buf145, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_14
        buf146 = reinterpret_tensor(buf141, (48, 64, 512), (32768, 512, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_223, reinterpret_tensor(buf145, (48, 512, 512), (262144, 512, 1), 0), out=buf146)
        del permute_223
        buf153 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf146, buf153, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf154 = reinterpret_tensor(buf146, (2048, 768), (768, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf153, permute_233, out=buf154)
        del permute_233
        buf147 = reinterpret_tensor(buf103, (48, 512, 64), (32768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf145, (48, 512, 512), (262144, 512, 1), 0), permute_224, out=buf147)
        del permute_224
        buf158 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf147, buf158, 1572864, grid=grid(1572864), stream=stream0)
        buf159 = reinterpret_tensor(buf147, (2048, 768), (768, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf158, permute_237, out=buf159)
        del permute_237
        buf139 = reinterpret_tensor(buf135, (1, 768, 16), (12288, 1, 768), 0); del buf135  # reuse
        buf167 = buf133; del buf133  # reuse
        buf169 = reinterpret_tensor(buf123, (768, 16), (1, 768), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf132, buf149, buf154, buf159, mul_64, buf139, buf167, buf169, 12288, 128, grid=grid(12288), stream=stream0)
        buf140 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf139, buf140, 768, 16, grid=grid(768), stream=stream0)
        buf150 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (768, 2048), (1, 768), 0), view_198, out=buf150)
        buf151 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf148, buf151, 12288, 128, grid=grid(12288), stream=stream0)
        buf152 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf151, buf152, 768, 16, grid=grid(768), stream=stream0)
        buf155 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (768, 2048), (1, 768), 0), view_198, out=buf155)
        buf156 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf153, buf156, 12288, 128, grid=grid(12288), stream=stream0)
        buf157 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf156, buf157, 768, 16, grid=grid(768), stream=stream0)
        buf160 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (768, 2048), (1, 768), 0), view_198, out=buf160)
        del view_198
        buf161 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf158, buf161, 12288, 128, grid=grid(12288), stream=stream0)
        buf162 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf161, buf162, 768, 16, grid=grid(768), stream=stream0)
        buf163 = buf132; del buf132  # reuse
        buf166 = reinterpret_tensor(buf158, (4, 512, 768), (393216, 768, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf163, buf149, buf154, buf159, primals_148, mul_64, div_34, buf166, 2048, 768, grid=grid(2048), stream=stream0)
        del div_34
        del mul_64
        del primals_148
        buf168 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf167, buf168, 768, 16, grid=grid(768), stream=stream0)
        buf170 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf169, buf170, 768, 16, grid=grid(768), stream=stream0)
        buf171 = reinterpret_tensor(buf125, (2048, 3072), (3072, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (2048, 768), (768, 1), 0), permute_241, out=buf171)
        del permute_241
        buf172 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (768, 2048), (1, 768), 0), view_196, out=buf172)
        del view_196
        buf175 = reinterpret_tensor(buf171, (4, 512, 3072), (1572864, 3072, 1), 0); del buf171  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf175, addmm_52, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_52
        buf176 = reinterpret_tensor(buf163, (2048, 768), (768, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (2048, 3072), (3072, 1), 0), permute_245, out=buf176)
        del permute_245
        buf173 = reinterpret_tensor(buf169, (1, 768, 16), (12288, 1, 768), 0); del buf169  # reuse
        buf183 = buf167; del buf167  # reuse
        buf185 = reinterpret_tensor(buf161, (768, 16), (1, 768), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf166, buf176, mul_59, buf173, buf183, buf185, 12288, 128, grid=grid(12288), stream=stream0)
        buf174 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf173, buf174, 768, 16, grid=grid(768), stream=stream0)
        buf177 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (3072, 2048), (1, 3072), 0), view_194, out=buf177)
        del view_194
        buf178 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf175, buf178, 49152, 128, grid=grid(49152), stream=stream0)
        buf179 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf178, buf179, 3072, 16, grid=grid(3072), stream=stream0)
        buf182 = reinterpret_tensor(buf159, (4, 512, 768), (393216, 768, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf166, buf176, primals_142, mul_59, div_35, buf182, 2048, 768, grid=grid(2048), stream=stream0)
        del div_35
        del mul_59
        del primals_142
        buf184 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf183, buf184, 768, 16, grid=grid(768), stream=stream0)
        buf186 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf185, buf186, 768, 16, grid=grid(768), stream=stream0)
        buf187 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (2048, 768), (768, 1), 0), permute_249, out=buf187)
        del permute_249
        buf188 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (768, 2048), (1, 768), 0), view_192, out=buf188)
        del view_192
        buf191 = reinterpret_tensor(buf166, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf187, buf191, 1572864, grid=grid(1572864), stream=stream0)
        buf192 = reinterpret_tensor(buf187, (48, 512, 64), (32768, 64, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_254, reinterpret_tensor(buf191, (48, 512, 64), (32768, 64, 1), 0), out=buf192)
        del permute_254
        buf198 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf192, buf198, 1572864, grid=grid(1572864), stream=stream0)
        buf199 = reinterpret_tensor(buf192, (2048, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf198, permute_261, out=buf199)
        del permute_261
        buf193 = reinterpret_tensor(buf145, (48, 512, 512), (262144, 512, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (48, 512, 64), (32768, 64, 1), 0), permute_255, out=buf193)
        del permute_255
        buf195 = reinterpret_tensor(buf143, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf193, alias_15, buf195, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_15
        buf196 = reinterpret_tensor(buf191, (48, 64, 512), (32768, 512, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_256, reinterpret_tensor(buf195, (48, 512, 512), (262144, 512, 1), 0), out=buf196)
        del permute_256
        buf203 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf196, buf203, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf204 = reinterpret_tensor(buf196, (2048, 768), (768, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf203, permute_266, out=buf204)
        del permute_266
        buf197 = reinterpret_tensor(buf153, (48, 512, 64), (32768, 64, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf195, (48, 512, 512), (262144, 512, 1), 0), permute_257, out=buf197)
        del permute_257
        buf208 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf197, buf208, 1572864, grid=grid(1572864), stream=stream0)
        buf209 = reinterpret_tensor(buf197, (2048, 768), (768, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf208, permute_270, out=buf209)
        del permute_270
        buf189 = reinterpret_tensor(buf185, (1, 768, 16), (12288, 1, 768), 0); del buf185  # reuse
        buf217 = buf183; del buf183  # reuse
        buf219 = reinterpret_tensor(buf173, (768, 16), (1, 768), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf182, buf199, buf204, buf209, mul_57, buf189, buf217, buf219, 12288, 128, grid=grid(12288), stream=stream0)
        buf190 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf189, buf190, 768, 16, grid=grid(768), stream=stream0)
        buf200 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (768, 2048), (1, 768), 0), view_176, out=buf200)
        buf201 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf198, buf201, 12288, 128, grid=grid(12288), stream=stream0)
        buf202 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf201, buf202, 768, 16, grid=grid(768), stream=stream0)
        buf205 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (768, 2048), (1, 768), 0), view_176, out=buf205)
        buf206 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf203, buf206, 12288, 128, grid=grid(12288), stream=stream0)
        buf207 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf206, buf207, 768, 16, grid=grid(768), stream=stream0)
        buf210 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (768, 2048), (1, 768), 0), view_176, out=buf210)
        del view_176
        buf211 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf208, buf211, 12288, 128, grid=grid(12288), stream=stream0)
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf211, buf212, 768, 16, grid=grid(768), stream=stream0)
        buf213 = buf182; del buf182  # reuse
        buf216 = reinterpret_tensor(buf208, (4, 512, 768), (393216, 768, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf213, buf199, buf204, buf209, primals_132, mul_57, div_37, buf216, 2048, 768, grid=grid(2048), stream=stream0)
        del div_37
        del mul_57
        del primals_132
        buf218 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf217, buf218, 768, 16, grid=grid(768), stream=stream0)
        buf220 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf219, buf220, 768, 16, grid=grid(768), stream=stream0)
        buf221 = reinterpret_tensor(buf175, (2048, 3072), (3072, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (2048, 768), (768, 1), 0), permute_274, out=buf221)
        del permute_274
        buf222 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (768, 2048), (1, 768), 0), view_174, out=buf222)
        del view_174
        buf225 = reinterpret_tensor(buf221, (4, 512, 3072), (1572864, 3072, 1), 0); del buf221  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf225, addmm_46, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_46
        buf226 = reinterpret_tensor(buf213, (2048, 768), (768, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (2048, 3072), (3072, 1), 0), permute_278, out=buf226)
        del permute_278
        buf223 = reinterpret_tensor(buf219, (1, 768, 16), (12288, 1, 768), 0); del buf219  # reuse
        buf233 = buf217; del buf217  # reuse
        buf235 = reinterpret_tensor(buf211, (768, 16), (1, 768), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf216, buf226, mul_52, buf223, buf233, buf235, 12288, 128, grid=grid(12288), stream=stream0)
        buf224 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf223, buf224, 768, 16, grid=grid(768), stream=stream0)
        buf227 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf225, (3072, 2048), (1, 3072), 0), view_172, out=buf227)
        del view_172
        buf228 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf225, buf228, 49152, 128, grid=grid(49152), stream=stream0)
        buf229 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf228, buf229, 3072, 16, grid=grid(3072), stream=stream0)
        buf232 = reinterpret_tensor(buf209, (4, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf216, buf226, primals_126, mul_52, div_38, buf232, 2048, 768, grid=grid(2048), stream=stream0)
        del div_38
        del mul_52
        del primals_126
        buf234 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf233, buf234, 768, 16, grid=grid(768), stream=stream0)
        buf236 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf235, buf236, 768, 16, grid=grid(768), stream=stream0)
        buf237 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (2048, 768), (768, 1), 0), permute_282, out=buf237)
        del permute_282
        buf238 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (768, 2048), (1, 768), 0), view_170, out=buf238)
        del view_170
        buf241 = reinterpret_tensor(buf216, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf237, buf241, 1572864, grid=grid(1572864), stream=stream0)
        buf242 = reinterpret_tensor(buf237, (48, 512, 64), (32768, 64, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_287, reinterpret_tensor(buf241, (48, 512, 64), (32768, 64, 1), 0), out=buf242)
        del permute_287
        buf248 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf242, buf248, 1572864, grid=grid(1572864), stream=stream0)
        buf249 = reinterpret_tensor(buf242, (2048, 768), (768, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf248, permute_294, out=buf249)
        del permute_294
        buf243 = reinterpret_tensor(buf195, (48, 512, 512), (262144, 512, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (48, 512, 64), (32768, 64, 1), 0), permute_288, out=buf243)
        del permute_288
        buf245 = reinterpret_tensor(buf193, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf243, alias_16, buf245, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_16
        buf246 = reinterpret_tensor(buf241, (48, 64, 512), (32768, 512, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_289, reinterpret_tensor(buf245, (48, 512, 512), (262144, 512, 1), 0), out=buf246)
        del permute_289
        buf253 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf246, buf253, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf254 = reinterpret_tensor(buf246, (2048, 768), (768, 1), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf253, permute_299, out=buf254)
        del permute_299
        buf247 = reinterpret_tensor(buf203, (48, 512, 64), (32768, 64, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (48, 512, 512), (262144, 512, 1), 0), permute_290, out=buf247)
        del permute_290
        buf258 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf247, buf258, 1572864, grid=grid(1572864), stream=stream0)
        buf259 = reinterpret_tensor(buf247, (2048, 768), (768, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf258, permute_303, out=buf259)
        del permute_303
        buf239 = reinterpret_tensor(buf235, (1, 768, 16), (12288, 1, 768), 0); del buf235  # reuse
        buf267 = buf233; del buf233  # reuse
        buf269 = reinterpret_tensor(buf223, (768, 16), (1, 768), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf232, buf249, buf254, buf259, mul_50, buf239, buf267, buf269, 12288, 128, grid=grid(12288), stream=stream0)
        buf240 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf239, buf240, 768, 16, grid=grid(768), stream=stream0)
        buf250 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (768, 2048), (1, 768), 0), view_154, out=buf250)
        buf251 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf248, buf251, 12288, 128, grid=grid(12288), stream=stream0)
        buf252 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf251, buf252, 768, 16, grid=grid(768), stream=stream0)
        buf255 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf253, (768, 2048), (1, 768), 0), view_154, out=buf255)
        buf256 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf253, buf256, 12288, 128, grid=grid(12288), stream=stream0)
        buf257 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf256, buf257, 768, 16, grid=grid(768), stream=stream0)
        buf260 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (768, 2048), (1, 768), 0), view_154, out=buf260)
        del view_154
        buf261 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf258, buf261, 12288, 128, grid=grid(12288), stream=stream0)
        buf262 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf261, buf262, 768, 16, grid=grid(768), stream=stream0)
        buf263 = buf232; del buf232  # reuse
        buf266 = reinterpret_tensor(buf258, (4, 512, 768), (393216, 768, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf263, buf249, buf254, buf259, primals_116, mul_50, div_40, buf266, 2048, 768, grid=grid(2048), stream=stream0)
        del div_40
        del mul_50
        del primals_116
        buf268 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf267, buf268, 768, 16, grid=grid(768), stream=stream0)
        buf270 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf269, buf270, 768, 16, grid=grid(768), stream=stream0)
        buf271 = reinterpret_tensor(buf225, (2048, 3072), (3072, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (2048, 768), (768, 1), 0), permute_307, out=buf271)
        del permute_307
        buf272 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (768, 2048), (1, 768), 0), view_152, out=buf272)
        del view_152
        buf275 = reinterpret_tensor(buf271, (4, 512, 3072), (1572864, 3072, 1), 0); del buf271  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf275, addmm_40, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_40
        buf276 = reinterpret_tensor(buf263, (2048, 768), (768, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (2048, 3072), (3072, 1), 0), permute_311, out=buf276)
        del permute_311
        buf273 = reinterpret_tensor(buf269, (1, 768, 16), (12288, 1, 768), 0); del buf269  # reuse
        buf283 = buf267; del buf267  # reuse
        buf285 = reinterpret_tensor(buf261, (768, 16), (1, 768), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf266, buf276, mul_45, buf273, buf283, buf285, 12288, 128, grid=grid(12288), stream=stream0)
        buf274 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf273, buf274, 768, 16, grid=grid(768), stream=stream0)
        buf277 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (3072, 2048), (1, 3072), 0), view_150, out=buf277)
        del view_150
        buf278 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf275, buf278, 49152, 128, grid=grid(49152), stream=stream0)
        buf279 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf278, buf279, 3072, 16, grid=grid(3072), stream=stream0)
        buf282 = reinterpret_tensor(buf259, (4, 512, 768), (393216, 768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf266, buf276, primals_110, mul_45, div_41, buf282, 2048, 768, grid=grid(2048), stream=stream0)
        del div_41
        del mul_45
        del primals_110
        buf284 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf283, buf284, 768, 16, grid=grid(768), stream=stream0)
        buf286 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf285, buf286, 768, 16, grid=grid(768), stream=stream0)
        buf287 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (2048, 768), (768, 1), 0), permute_315, out=buf287)
        del permute_315
        buf288 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (768, 2048), (1, 768), 0), view_148, out=buf288)
        del view_148
        buf291 = reinterpret_tensor(buf266, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf287, buf291, 1572864, grid=grid(1572864), stream=stream0)
        buf292 = reinterpret_tensor(buf287, (48, 512, 64), (32768, 64, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_320, reinterpret_tensor(buf291, (48, 512, 64), (32768, 64, 1), 0), out=buf292)
        del permute_320
        buf298 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf292, buf298, 1572864, grid=grid(1572864), stream=stream0)
        buf299 = reinterpret_tensor(buf292, (2048, 768), (768, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf298, permute_327, out=buf299)
        del permute_327
        buf293 = reinterpret_tensor(buf245, (48, 512, 512), (262144, 512, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (48, 512, 64), (32768, 64, 1), 0), permute_321, out=buf293)
        del permute_321
        buf295 = reinterpret_tensor(buf243, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf293, alias_17, buf295, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_17
        buf296 = reinterpret_tensor(buf291, (48, 64, 512), (32768, 512, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_322, reinterpret_tensor(buf295, (48, 512, 512), (262144, 512, 1), 0), out=buf296)
        del permute_322
        buf303 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf296, buf303, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf304 = reinterpret_tensor(buf296, (2048, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf303, permute_332, out=buf304)
        del permute_332
        buf297 = reinterpret_tensor(buf253, (48, 512, 64), (32768, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf295, (48, 512, 512), (262144, 512, 1), 0), permute_323, out=buf297)
        del permute_323
        buf308 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf297, buf308, 1572864, grid=grid(1572864), stream=stream0)
        buf309 = reinterpret_tensor(buf297, (2048, 768), (768, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf308, permute_336, out=buf309)
        del permute_336
        buf289 = reinterpret_tensor(buf285, (1, 768, 16), (12288, 1, 768), 0); del buf285  # reuse
        buf317 = buf283; del buf283  # reuse
        buf319 = reinterpret_tensor(buf273, (768, 16), (1, 768), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf282, buf299, buf304, buf309, mul_43, buf289, buf317, buf319, 12288, 128, grid=grid(12288), stream=stream0)
        buf290 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf289, buf290, 768, 16, grid=grid(768), stream=stream0)
        buf300 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (768, 2048), (1, 768), 0), view_132, out=buf300)
        buf301 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf298, buf301, 12288, 128, grid=grid(12288), stream=stream0)
        buf302 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf301, buf302, 768, 16, grid=grid(768), stream=stream0)
        buf305 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (768, 2048), (1, 768), 0), view_132, out=buf305)
        buf306 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf303, buf306, 12288, 128, grid=grid(12288), stream=stream0)
        buf307 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf306, buf307, 768, 16, grid=grid(768), stream=stream0)
        buf310 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (768, 2048), (1, 768), 0), view_132, out=buf310)
        del view_132
        buf311 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf308, buf311, 12288, 128, grid=grid(12288), stream=stream0)
        buf312 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf311, buf312, 768, 16, grid=grid(768), stream=stream0)
        buf313 = buf282; del buf282  # reuse
        buf316 = reinterpret_tensor(buf308, (4, 512, 768), (393216, 768, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf313, buf299, buf304, buf309, primals_100, mul_43, div_43, buf316, 2048, 768, grid=grid(2048), stream=stream0)
        del div_43
        del mul_43
        del primals_100
        buf318 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf317, buf318, 768, 16, grid=grid(768), stream=stream0)
        buf320 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf319, buf320, 768, 16, grid=grid(768), stream=stream0)
        buf321 = reinterpret_tensor(buf275, (2048, 3072), (3072, 1), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (2048, 768), (768, 1), 0), permute_340, out=buf321)
        del permute_340
        buf322 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (768, 2048), (1, 768), 0), view_130, out=buf322)
        del view_130
        buf325 = reinterpret_tensor(buf321, (4, 512, 3072), (1572864, 3072, 1), 0); del buf321  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf325, addmm_34, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_34
        buf326 = reinterpret_tensor(buf313, (2048, 768), (768, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (2048, 3072), (3072, 1), 0), permute_344, out=buf326)
        del permute_344
        buf323 = reinterpret_tensor(buf319, (1, 768, 16), (12288, 1, 768), 0); del buf319  # reuse
        buf333 = buf317; del buf317  # reuse
        buf335 = reinterpret_tensor(buf311, (768, 16), (1, 768), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf316, buf326, mul_38, buf323, buf333, buf335, 12288, 128, grid=grid(12288), stream=stream0)
        buf324 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf323, buf324, 768, 16, grid=grid(768), stream=stream0)
        buf327 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (3072, 2048), (1, 3072), 0), view_128, out=buf327)
        del view_128
        buf328 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf325, buf328, 49152, 128, grid=grid(49152), stream=stream0)
        buf329 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf328, buf329, 3072, 16, grid=grid(3072), stream=stream0)
        buf332 = reinterpret_tensor(buf309, (4, 512, 768), (393216, 768, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf316, buf326, primals_94, mul_38, div_44, buf332, 2048, 768, grid=grid(2048), stream=stream0)
        del div_44
        del mul_38
        del primals_94
        buf334 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf333, buf334, 768, 16, grid=grid(768), stream=stream0)
        buf336 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf335, buf336, 768, 16, grid=grid(768), stream=stream0)
        buf337 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (2048, 768), (768, 1), 0), permute_348, out=buf337)
        del permute_348
        buf338 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 2048), (1, 768), 0), view_126, out=buf338)
        del view_126
        buf341 = reinterpret_tensor(buf316, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf337, buf341, 1572864, grid=grid(1572864), stream=stream0)
        buf342 = reinterpret_tensor(buf337, (48, 512, 64), (32768, 64, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_353, reinterpret_tensor(buf341, (48, 512, 64), (32768, 64, 1), 0), out=buf342)
        del permute_353
        buf348 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf342, buf348, 1572864, grid=grid(1572864), stream=stream0)
        buf349 = reinterpret_tensor(buf342, (2048, 768), (768, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf348, permute_360, out=buf349)
        del permute_360
        buf343 = reinterpret_tensor(buf295, (48, 512, 512), (262144, 512, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (48, 512, 64), (32768, 64, 1), 0), permute_354, out=buf343)
        del permute_354
        buf345 = reinterpret_tensor(buf293, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf343, alias_18, buf345, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_18
        buf346 = reinterpret_tensor(buf341, (48, 64, 512), (32768, 512, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_355, reinterpret_tensor(buf345, (48, 512, 512), (262144, 512, 1), 0), out=buf346)
        del permute_355
        buf353 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf346, buf353, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf354 = reinterpret_tensor(buf346, (2048, 768), (768, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf353, permute_365, out=buf354)
        del permute_365
        buf347 = reinterpret_tensor(buf303, (48, 512, 64), (32768, 64, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf345, (48, 512, 512), (262144, 512, 1), 0), permute_356, out=buf347)
        del permute_356
        buf358 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf347, buf358, 1572864, grid=grid(1572864), stream=stream0)
        buf359 = reinterpret_tensor(buf347, (2048, 768), (768, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf358, permute_369, out=buf359)
        del permute_369
        buf339 = reinterpret_tensor(buf335, (1, 768, 16), (12288, 1, 768), 0); del buf335  # reuse
        buf367 = buf333; del buf333  # reuse
        buf369 = reinterpret_tensor(buf323, (768, 16), (1, 768), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf332, buf349, buf354, buf359, mul_36, buf339, buf367, buf369, 12288, 128, grid=grid(12288), stream=stream0)
        buf340 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf339, buf340, 768, 16, grid=grid(768), stream=stream0)
        buf350 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (768, 2048), (1, 768), 0), view_110, out=buf350)
        buf351 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf348, buf351, 12288, 128, grid=grid(12288), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf351, buf352, 768, 16, grid=grid(768), stream=stream0)
        buf355 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (768, 2048), (1, 768), 0), view_110, out=buf355)
        buf356 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf353, buf356, 12288, 128, grid=grid(12288), stream=stream0)
        buf357 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf356, buf357, 768, 16, grid=grid(768), stream=stream0)
        buf360 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (768, 2048), (1, 768), 0), view_110, out=buf360)
        del view_110
        buf361 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf358, buf361, 12288, 128, grid=grid(12288), stream=stream0)
        buf362 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf361, buf362, 768, 16, grid=grid(768), stream=stream0)
        buf363 = buf332; del buf332  # reuse
        buf366 = reinterpret_tensor(buf358, (4, 512, 768), (393216, 768, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf363, buf349, buf354, buf359, primals_84, mul_36, div_46, buf366, 2048, 768, grid=grid(2048), stream=stream0)
        del div_46
        del mul_36
        del primals_84
        buf368 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf367, buf368, 768, 16, grid=grid(768), stream=stream0)
        buf370 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf369, buf370, 768, 16, grid=grid(768), stream=stream0)
        buf371 = reinterpret_tensor(buf325, (2048, 3072), (3072, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (2048, 768), (768, 1), 0), permute_373, out=buf371)
        del permute_373
        buf372 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (768, 2048), (1, 768), 0), view_108, out=buf372)
        del view_108
        buf375 = reinterpret_tensor(buf371, (4, 512, 3072), (1572864, 3072, 1), 0); del buf371  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf375, addmm_28, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_28
        buf376 = reinterpret_tensor(buf363, (2048, 768), (768, 1), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (2048, 3072), (3072, 1), 0), permute_377, out=buf376)
        del permute_377
        buf373 = reinterpret_tensor(buf369, (1, 768, 16), (12288, 1, 768), 0); del buf369  # reuse
        buf383 = buf367; del buf367  # reuse
        buf385 = reinterpret_tensor(buf361, (768, 16), (1, 768), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf366, buf376, mul_31, buf373, buf383, buf385, 12288, 128, grid=grid(12288), stream=stream0)
        buf374 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf373, buf374, 768, 16, grid=grid(768), stream=stream0)
        buf377 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (3072, 2048), (1, 3072), 0), view_106, out=buf377)
        del view_106
        buf378 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf375, buf378, 49152, 128, grid=grid(49152), stream=stream0)
        buf379 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf378, buf379, 3072, 16, grid=grid(3072), stream=stream0)
        buf382 = reinterpret_tensor(buf359, (4, 512, 768), (393216, 768, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf366, buf376, primals_78, mul_31, div_47, buf382, 2048, 768, grid=grid(2048), stream=stream0)
        del div_47
        del mul_31
        del primals_78
        buf384 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf383, buf384, 768, 16, grid=grid(768), stream=stream0)
        buf386 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf385, buf386, 768, 16, grid=grid(768), stream=stream0)
        buf387 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (2048, 768), (768, 1), 0), permute_381, out=buf387)
        del permute_381
        buf388 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (768, 2048), (1, 768), 0), view_104, out=buf388)
        del view_104
        buf391 = reinterpret_tensor(buf366, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf387, buf391, 1572864, grid=grid(1572864), stream=stream0)
        buf392 = reinterpret_tensor(buf387, (48, 512, 64), (32768, 64, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_386, reinterpret_tensor(buf391, (48, 512, 64), (32768, 64, 1), 0), out=buf392)
        del permute_386
        buf398 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf392, buf398, 1572864, grid=grid(1572864), stream=stream0)
        buf399 = reinterpret_tensor(buf392, (2048, 768), (768, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf398, permute_393, out=buf399)
        del permute_393
        buf393 = reinterpret_tensor(buf345, (48, 512, 512), (262144, 512, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (48, 512, 64), (32768, 64, 1), 0), permute_387, out=buf393)
        del permute_387
        buf395 = reinterpret_tensor(buf343, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf393, alias_19, buf395, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_19
        buf396 = reinterpret_tensor(buf391, (48, 64, 512), (32768, 512, 1), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_388, reinterpret_tensor(buf395, (48, 512, 512), (262144, 512, 1), 0), out=buf396)
        del permute_388
        buf403 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf396, buf403, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf404 = reinterpret_tensor(buf396, (2048, 768), (768, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf403, permute_398, out=buf404)
        del permute_398
        buf397 = reinterpret_tensor(buf353, (48, 512, 64), (32768, 64, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (48, 512, 512), (262144, 512, 1), 0), permute_389, out=buf397)
        del permute_389
        buf408 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf397, buf408, 1572864, grid=grid(1572864), stream=stream0)
        buf409 = reinterpret_tensor(buf397, (2048, 768), (768, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf408, permute_402, out=buf409)
        del permute_402
        buf389 = reinterpret_tensor(buf385, (1, 768, 16), (12288, 1, 768), 0); del buf385  # reuse
        buf417 = buf383; del buf383  # reuse
        buf419 = reinterpret_tensor(buf373, (768, 16), (1, 768), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf382, buf399, buf404, buf409, mul_29, buf389, buf417, buf419, 12288, 128, grid=grid(12288), stream=stream0)
        buf390 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf389, buf390, 768, 16, grid=grid(768), stream=stream0)
        buf400 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (768, 2048), (1, 768), 0), view_88, out=buf400)
        buf401 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf398, buf401, 12288, 128, grid=grid(12288), stream=stream0)
        buf402 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf401, buf402, 768, 16, grid=grid(768), stream=stream0)
        buf405 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (768, 2048), (1, 768), 0), view_88, out=buf405)
        buf406 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf403, buf406, 12288, 128, grid=grid(12288), stream=stream0)
        buf407 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf406, buf407, 768, 16, grid=grid(768), stream=stream0)
        buf410 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (768, 2048), (1, 768), 0), view_88, out=buf410)
        del view_88
        buf411 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf408, buf411, 12288, 128, grid=grid(12288), stream=stream0)
        buf412 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf411, buf412, 768, 16, grid=grid(768), stream=stream0)
        buf413 = buf382; del buf382  # reuse
        buf416 = reinterpret_tensor(buf408, (4, 512, 768), (393216, 768, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf413, buf399, buf404, buf409, primals_68, mul_29, div_49, buf416, 2048, 768, grid=grid(2048), stream=stream0)
        del div_49
        del mul_29
        del primals_68
        buf418 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf417, buf418, 768, 16, grid=grid(768), stream=stream0)
        buf420 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf419, buf420, 768, 16, grid=grid(768), stream=stream0)
        buf421 = reinterpret_tensor(buf375, (2048, 3072), (3072, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (2048, 768), (768, 1), 0), permute_406, out=buf421)
        del permute_406
        buf422 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (768, 2048), (1, 768), 0), view_86, out=buf422)
        del view_86
        buf425 = reinterpret_tensor(buf421, (4, 512, 3072), (1572864, 3072, 1), 0); del buf421  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf425, addmm_22, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_22
        buf426 = reinterpret_tensor(buf413, (2048, 768), (768, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (2048, 3072), (3072, 1), 0), permute_410, out=buf426)
        del permute_410
        buf423 = reinterpret_tensor(buf419, (1, 768, 16), (12288, 1, 768), 0); del buf419  # reuse
        buf433 = buf417; del buf417  # reuse
        buf435 = reinterpret_tensor(buf411, (768, 16), (1, 768), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf416, buf426, mul_24, buf423, buf433, buf435, 12288, 128, grid=grid(12288), stream=stream0)
        buf424 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf423, buf424, 768, 16, grid=grid(768), stream=stream0)
        buf427 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (3072, 2048), (1, 3072), 0), view_84, out=buf427)
        del view_84
        buf428 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf425, buf428, 49152, 128, grid=grid(49152), stream=stream0)
        buf429 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf428, buf429, 3072, 16, grid=grid(3072), stream=stream0)
        buf432 = reinterpret_tensor(buf409, (4, 512, 768), (393216, 768, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf416, buf426, primals_62, mul_24, div_50, buf432, 2048, 768, grid=grid(2048), stream=stream0)
        del div_50
        del mul_24
        del primals_62
        buf434 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf433, buf434, 768, 16, grid=grid(768), stream=stream0)
        buf436 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf435, buf436, 768, 16, grid=grid(768), stream=stream0)
        buf437 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (2048, 768), (768, 1), 0), permute_414, out=buf437)
        del permute_414
        buf438 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (768, 2048), (1, 768), 0), view_82, out=buf438)
        del view_82
        buf441 = reinterpret_tensor(buf416, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf437, buf441, 1572864, grid=grid(1572864), stream=stream0)
        buf442 = reinterpret_tensor(buf437, (48, 512, 64), (32768, 64, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_419, reinterpret_tensor(buf441, (48, 512, 64), (32768, 64, 1), 0), out=buf442)
        del permute_419
        buf448 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf442, buf448, 1572864, grid=grid(1572864), stream=stream0)
        buf449 = reinterpret_tensor(buf442, (2048, 768), (768, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf448, permute_426, out=buf449)
        del permute_426
        buf443 = reinterpret_tensor(buf395, (48, 512, 512), (262144, 512, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (48, 512, 64), (32768, 64, 1), 0), permute_420, out=buf443)
        del permute_420
        buf445 = reinterpret_tensor(buf393, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf443, alias_20, buf445, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_20
        buf446 = reinterpret_tensor(buf441, (48, 64, 512), (32768, 512, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_421, reinterpret_tensor(buf445, (48, 512, 512), (262144, 512, 1), 0), out=buf446)
        del permute_421
        buf453 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf446, buf453, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf454 = reinterpret_tensor(buf446, (2048, 768), (768, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf453, permute_431, out=buf454)
        del permute_431
        buf447 = reinterpret_tensor(buf403, (48, 512, 64), (32768, 64, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (48, 512, 512), (262144, 512, 1), 0), permute_422, out=buf447)
        del permute_422
        buf458 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf447, buf458, 1572864, grid=grid(1572864), stream=stream0)
        buf459 = reinterpret_tensor(buf447, (2048, 768), (768, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf458, permute_435, out=buf459)
        del permute_435
        buf439 = reinterpret_tensor(buf435, (1, 768, 16), (12288, 1, 768), 0); del buf435  # reuse
        buf467 = buf433; del buf433  # reuse
        buf469 = reinterpret_tensor(buf423, (768, 16), (1, 768), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf432, buf449, buf454, buf459, mul_22, buf439, buf467, buf469, 12288, 128, grid=grid(12288), stream=stream0)
        buf440 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf439, buf440, 768, 16, grid=grid(768), stream=stream0)
        buf450 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf448, (768, 2048), (1, 768), 0), view_66, out=buf450)
        buf451 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf448, buf451, 12288, 128, grid=grid(12288), stream=stream0)
        buf452 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf451, buf452, 768, 16, grid=grid(768), stream=stream0)
        buf455 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (768, 2048), (1, 768), 0), view_66, out=buf455)
        buf456 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf453, buf456, 12288, 128, grid=grid(12288), stream=stream0)
        buf457 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf456, buf457, 768, 16, grid=grid(768), stream=stream0)
        buf460 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (768, 2048), (1, 768), 0), view_66, out=buf460)
        del view_66
        buf461 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf458, buf461, 12288, 128, grid=grid(12288), stream=stream0)
        buf462 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf461, buf462, 768, 16, grid=grid(768), stream=stream0)
        buf463 = buf432; del buf432  # reuse
        buf466 = reinterpret_tensor(buf458, (4, 512, 768), (393216, 768, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf463, buf449, buf454, buf459, primals_52, mul_22, div_52, buf466, 2048, 768, grid=grid(2048), stream=stream0)
        del div_52
        del mul_22
        del primals_52
        buf468 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf467, buf468, 768, 16, grid=grid(768), stream=stream0)
        buf470 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf469, buf470, 768, 16, grid=grid(768), stream=stream0)
        buf471 = reinterpret_tensor(buf425, (2048, 3072), (3072, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (2048, 768), (768, 1), 0), permute_439, out=buf471)
        del permute_439
        buf472 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (768, 2048), (1, 768), 0), view_64, out=buf472)
        del view_64
        buf475 = reinterpret_tensor(buf471, (4, 512, 3072), (1572864, 3072, 1), 0); del buf471  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf475, addmm_16, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_16
        buf476 = reinterpret_tensor(buf463, (2048, 768), (768, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (2048, 3072), (3072, 1), 0), permute_443, out=buf476)
        del permute_443
        buf473 = reinterpret_tensor(buf469, (1, 768, 16), (12288, 1, 768), 0); del buf469  # reuse
        buf483 = buf467; del buf467  # reuse
        buf485 = reinterpret_tensor(buf461, (768, 16), (1, 768), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf466, buf476, mul_17, buf473, buf483, buf485, 12288, 128, grid=grid(12288), stream=stream0)
        buf474 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf473, buf474, 768, 16, grid=grid(768), stream=stream0)
        buf477 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf475, (3072, 2048), (1, 3072), 0), view_62, out=buf477)
        del view_62
        buf478 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf475, buf478, 49152, 128, grid=grid(49152), stream=stream0)
        buf479 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf478, buf479, 3072, 16, grid=grid(3072), stream=stream0)
        buf482 = reinterpret_tensor(buf459, (4, 512, 768), (393216, 768, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf466, buf476, primals_46, mul_17, div_53, buf482, 2048, 768, grid=grid(2048), stream=stream0)
        del div_53
        del mul_17
        del primals_46
        buf484 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf483, buf484, 768, 16, grid=grid(768), stream=stream0)
        buf486 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf485, buf486, 768, 16, grid=grid(768), stream=stream0)
        buf487 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (2048, 768), (768, 1), 0), permute_447, out=buf487)
        del permute_447
        buf488 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (768, 2048), (1, 768), 0), view_60, out=buf488)
        del view_60
        buf491 = reinterpret_tensor(buf466, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf487, buf491, 1572864, grid=grid(1572864), stream=stream0)
        buf492 = reinterpret_tensor(buf487, (48, 512, 64), (32768, 64, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_452, reinterpret_tensor(buf491, (48, 512, 64), (32768, 64, 1), 0), out=buf492)
        del permute_452
        buf498 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf492, buf498, 1572864, grid=grid(1572864), stream=stream0)
        buf499 = reinterpret_tensor(buf492, (2048, 768), (768, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf498, permute_459, out=buf499)
        del permute_459
        buf493 = reinterpret_tensor(buf445, (48, 512, 512), (262144, 512, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf491, (48, 512, 64), (32768, 64, 1), 0), permute_453, out=buf493)
        del permute_453
        buf495 = reinterpret_tensor(buf443, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf493, alias_21, buf495, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_21
        buf496 = reinterpret_tensor(buf491, (48, 64, 512), (32768, 512, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_454, reinterpret_tensor(buf495, (48, 512, 512), (262144, 512, 1), 0), out=buf496)
        del permute_454
        buf503 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf496, buf503, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf504 = reinterpret_tensor(buf496, (2048, 768), (768, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf503, permute_464, out=buf504)
        del permute_464
        buf497 = reinterpret_tensor(buf453, (48, 512, 64), (32768, 64, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (48, 512, 512), (262144, 512, 1), 0), permute_455, out=buf497)
        del permute_455
        buf508 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf497, buf508, 1572864, grid=grid(1572864), stream=stream0)
        buf509 = reinterpret_tensor(buf497, (2048, 768), (768, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf508, permute_468, out=buf509)
        del permute_468
        buf489 = reinterpret_tensor(buf485, (1, 768, 16), (12288, 1, 768), 0); del buf485  # reuse
        buf517 = buf483; del buf483  # reuse
        buf519 = reinterpret_tensor(buf473, (768, 16), (1, 768), 0); del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf482, buf499, buf504, buf509, mul_15, buf489, buf517, buf519, 12288, 128, grid=grid(12288), stream=stream0)
        buf490 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf489, buf490, 768, 16, grid=grid(768), stream=stream0)
        buf500 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (768, 2048), (1, 768), 0), view_44, out=buf500)
        buf501 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf498, buf501, 12288, 128, grid=grid(12288), stream=stream0)
        buf502 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf501, buf502, 768, 16, grid=grid(768), stream=stream0)
        buf505 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf503, (768, 2048), (1, 768), 0), view_44, out=buf505)
        buf506 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf503, buf506, 12288, 128, grid=grid(12288), stream=stream0)
        buf507 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf506, buf507, 768, 16, grid=grid(768), stream=stream0)
        buf510 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (768, 2048), (1, 768), 0), view_44, out=buf510)
        del view_44
        buf511 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf508, buf511, 12288, 128, grid=grid(12288), stream=stream0)
        buf512 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf511, buf512, 768, 16, grid=grid(768), stream=stream0)
        buf513 = buf482; del buf482  # reuse
        buf516 = reinterpret_tensor(buf508, (4, 512, 768), (393216, 768, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf513, buf499, buf504, buf509, primals_36, mul_15, div_55, buf516, 2048, 768, grid=grid(2048), stream=stream0)
        del div_55
        del mul_15
        del primals_36
        buf518 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf517, buf518, 768, 16, grid=grid(768), stream=stream0)
        buf520 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf519, buf520, 768, 16, grid=grid(768), stream=stream0)
        buf521 = reinterpret_tensor(buf475, (2048, 3072), (3072, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (2048, 768), (768, 1), 0), permute_472, out=buf521)
        del permute_472
        buf522 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf516, (768, 2048), (1, 768), 0), view_42, out=buf522)
        del view_42
        buf525 = reinterpret_tensor(buf521, (4, 512, 3072), (1572864, 3072, 1), 0); del buf521  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf525, addmm_10, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_10
        buf526 = reinterpret_tensor(buf513, (2048, 768), (768, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (2048, 3072), (3072, 1), 0), permute_476, out=buf526)
        del permute_476
        buf523 = reinterpret_tensor(buf519, (1, 768, 16), (12288, 1, 768), 0); del buf519  # reuse
        buf533 = buf517; del buf517  # reuse
        buf535 = reinterpret_tensor(buf511, (768, 16), (1, 768), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf516, buf526, mul_10, buf523, buf533, buf535, 12288, 128, grid=grid(12288), stream=stream0)
        buf524 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf523, buf524, 768, 16, grid=grid(768), stream=stream0)
        buf527 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (3072, 2048), (1, 3072), 0), view_40, out=buf527)
        del view_40
        buf528 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf525, buf528, 49152, 128, grid=grid(49152), stream=stream0)
        buf529 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf528, buf529, 3072, 16, grid=grid(3072), stream=stream0)
        buf532 = reinterpret_tensor(buf509, (4, 512, 768), (393216, 768, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf516, buf526, primals_30, mul_10, div_56, buf532, 2048, 768, grid=grid(2048), stream=stream0)
        del div_56
        del mul_10
        del primals_30
        buf534 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf533, buf534, 768, 16, grid=grid(768), stream=stream0)
        buf536 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf535, buf536, 768, 16, grid=grid(768), stream=stream0)
        buf537 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (2048, 768), (768, 1), 0), permute_480, out=buf537)
        del permute_480
        buf538 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (768, 2048), (1, 768), 0), view_38, out=buf538)
        del view_38
        buf541 = reinterpret_tensor(buf516, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf537, buf541, 1572864, grid=grid(1572864), stream=stream0)
        buf542 = reinterpret_tensor(buf537, (48, 512, 64), (32768, 64, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_485, reinterpret_tensor(buf541, (48, 512, 64), (32768, 64, 1), 0), out=buf542)
        del permute_485
        buf548 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf542, buf548, 1572864, grid=grid(1572864), stream=stream0)
        buf549 = reinterpret_tensor(buf542, (2048, 768), (768, 1), 0); del buf542  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf548, permute_492, out=buf549)
        del permute_492
        buf543 = reinterpret_tensor(buf495, (48, 512, 512), (262144, 512, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf541, (48, 512, 64), (32768, 64, 1), 0), permute_486, out=buf543)
        del permute_486
        buf545 = reinterpret_tensor(buf493, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf543, alias_22, buf545, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_22
        buf546 = reinterpret_tensor(buf541, (48, 64, 512), (32768, 512, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_487, reinterpret_tensor(buf545, (48, 512, 512), (262144, 512, 1), 0), out=buf546)
        del permute_487
        buf553 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf546, buf553, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf554 = reinterpret_tensor(buf546, (2048, 768), (768, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf553, permute_497, out=buf554)
        del permute_497
        buf547 = reinterpret_tensor(buf503, (48, 512, 64), (32768, 64, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf545, (48, 512, 512), (262144, 512, 1), 0), permute_488, out=buf547)
        del permute_488
        buf558 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf547, buf558, 1572864, grid=grid(1572864), stream=stream0)
        buf559 = reinterpret_tensor(buf547, (2048, 768), (768, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf558, permute_501, out=buf559)
        del permute_501
        buf539 = reinterpret_tensor(buf535, (1, 768, 16), (12288, 1, 768), 0); del buf535  # reuse
        buf567 = buf533; del buf533  # reuse
        buf569 = reinterpret_tensor(buf523, (768, 16), (1, 768), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf532, buf549, buf554, buf559, mul_8, buf539, buf567, buf569, 12288, 128, grid=grid(12288), stream=stream0)
        buf540 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf539, buf540, 768, 16, grid=grid(768), stream=stream0)
        buf550 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (768, 2048), (1, 768), 0), view_22, out=buf550)
        buf551 = buf539; del buf539  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf548, buf551, 12288, 128, grid=grid(12288), stream=stream0)
        buf552 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf551, buf552, 768, 16, grid=grid(768), stream=stream0)
        buf555 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf553, (768, 2048), (1, 768), 0), view_22, out=buf555)
        buf556 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf553, buf556, 12288, 128, grid=grid(12288), stream=stream0)
        buf557 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf556, buf557, 768, 16, grid=grid(768), stream=stream0)
        buf560 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (768, 2048), (1, 768), 0), view_22, out=buf560)
        del view_22
        buf561 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf558, buf561, 12288, 128, grid=grid(12288), stream=stream0)
        buf562 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf561, buf562, 768, 16, grid=grid(768), stream=stream0)
        buf563 = buf532; del buf532  # reuse
        buf566 = reinterpret_tensor(buf558, (4, 512, 768), (393216, 768, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_16.run(buf563, buf549, buf554, buf559, primals_20, mul_8, div_58, buf566, 2048, 768, grid=grid(2048), stream=stream0)
        del div_58
        del mul_8
        del primals_20
        buf568 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf567, buf568, 768, 16, grid=grid(768), stream=stream0)
        buf570 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf569, buf570, 768, 16, grid=grid(768), stream=stream0)
        buf571 = reinterpret_tensor(buf525, (2048, 3072), (3072, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (2048, 768), (768, 1), 0), permute_505, out=buf571)
        del permute_505
        buf572 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (768, 2048), (1, 768), 0), view_20, out=buf572)
        del view_20
        buf575 = reinterpret_tensor(buf571, (4, 512, 3072), (1572864, 3072, 1), 0); del buf571  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf575, addmm_4, 6291456, grid=grid(6291456), stream=stream0)
        del addmm_4
        buf576 = reinterpret_tensor(buf563, (2048, 768), (768, 1), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (2048, 3072), (3072, 1), 0), permute_509, out=buf576)
        del permute_509
        buf573 = reinterpret_tensor(buf569, (1, 768, 16), (12288, 1, 768), 0); del buf569  # reuse
        buf583 = buf567; del buf567  # reuse
        buf585 = reinterpret_tensor(buf561, (768, 16), (1, 768), 0); del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_7.run(buf566, buf576, mul_3, buf573, buf583, buf585, 12288, 128, grid=grid(12288), stream=stream0)
        buf574 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf573, buf574, 768, 16, grid=grid(768), stream=stream0)
        buf577 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (3072, 2048), (1, 3072), 0), view_18, out=buf577)
        del view_18
        buf578 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf575, buf578, 49152, 128, grid=grid(49152), stream=stream0)
        del buf575
        buf579 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf578, buf579, 3072, 16, grid=grid(3072), stream=stream0)
        del buf578
        buf582 = reinterpret_tensor(buf559, (4, 512, 768), (393216, 768, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf566, buf576, primals_14, mul_3, div_59, buf582, 2048, 768, grid=grid(2048), stream=stream0)
        del div_59
        del mul_3
        del primals_14
        buf584 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf583, buf584, 768, 16, grid=grid(768), stream=stream0)
        buf586 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf585, buf586, 768, 16, grid=grid(768), stream=stream0)
        buf587 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf582, (2048, 768), (768, 1), 0), permute_513, out=buf587)
        del permute_513
        buf588 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf582, (768, 2048), (1, 768), 0), view_16, out=buf588)
        del view_16
        buf591 = reinterpret_tensor(buf566, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf587, buf591, 1572864, grid=grid(1572864), stream=stream0)
        buf592 = reinterpret_tensor(buf587, (48, 512, 64), (32768, 64, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_518, reinterpret_tensor(buf591, (48, 512, 64), (32768, 64, 1), 0), out=buf592)
        del permute_518
        buf598 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf592, buf598, 1572864, grid=grid(1572864), stream=stream0)
        buf599 = reinterpret_tensor(buf592, (2048, 768), (768, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf598, permute_525, out=buf599)
        del permute_525
        buf593 = reinterpret_tensor(buf545, (48, 512, 512), (262144, 512, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf591, (48, 512, 64), (32768, 64, 1), 0), permute_519, out=buf593)
        del permute_519
        buf595 = reinterpret_tensor(buf543, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div]
        triton_per_fused__softmax_backward_data_div_13.run(buf593, alias_23, buf595, 24576, 512, grid=grid(24576), stream=stream0)
        del alias_23
        del buf593
        buf596 = reinterpret_tensor(buf591, (48, 64, 512), (32768, 512, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_520, reinterpret_tensor(buf595, (48, 512, 512), (262144, 512, 1), 0), out=buf596)
        del permute_520
        buf603 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_14.run(buf596, buf603, 2048, 768, grid=grid(2048, 768), stream=stream0)
        buf604 = reinterpret_tensor(buf596, (2048, 768), (768, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf603, permute_530, out=buf604)
        del permute_530
        buf597 = reinterpret_tensor(buf553, (48, 512, 64), (32768, 64, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf595, (48, 512, 512), (262144, 512, 1), 0), permute_521, out=buf597)
        del buf595
        del permute_521
        buf608 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_12.run(buf597, buf608, 1572864, grid=grid(1572864), stream=stream0)
        buf609 = reinterpret_tensor(buf597, (2048, 768), (768, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf608, permute_534, out=buf609)
        del permute_534
        buf589 = reinterpret_tensor(buf585, (1, 768, 16), (12288, 1, 768), 0); del buf585  # reuse
        buf617 = buf583; del buf583  # reuse
        buf619 = reinterpret_tensor(buf573, (768, 16), (1, 768), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]
        triton_red_fused_add_native_layer_norm_backward_sum_15.run(buf582, buf599, buf604, buf609, mul_1, buf589, buf617, buf619, 12288, 128, grid=grid(12288), stream=stream0)
        buf590 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf589, buf590, 768, 16, grid=grid(768), stream=stream0)
        buf600 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf598, (768, 2048), (1, 768), 0), view, out=buf600)
        buf601 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf598, buf601, 12288, 128, grid=grid(12288), stream=stream0)
        buf602 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf601, buf602, 768, 16, grid=grid(768), stream=stream0)
        buf605 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (768, 2048), (1, 768), 0), view, out=buf605)
        buf606 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf603, buf606, 12288, 128, grid=grid(12288), stream=stream0)
        buf607 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf606, buf607, 768, 16, grid=grid(768), stream=stream0)
        buf610 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (768, 2048), (1, 768), 0), view, out=buf610)
        del view
        buf611 = buf606; del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf608, buf611, 12288, 128, grid=grid(12288), stream=stream0)
        buf612 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_3.run(buf611, buf612, 768, 16, grid=grid(768), stream=stream0)
        del buf611
        buf613 = buf582; del buf582  # reuse
        buf616 = reinterpret_tensor(buf608, (4, 512, 768), (393216, 768, 1), 0); del buf608  # reuse
        buf626 = reinterpret_tensor(buf603, (4, 512, 768), (393216, 768, 1), 0); del buf603  # reuse
        buf630 = reinterpret_tensor(buf598, (4, 512, 768), (393216, 768, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_18.run(buf613, buf599, buf604, buf609, primals_4, mul_1, div_61, expand, primals_206, buf616, buf626, buf630, 2048, 768, grid=grid(2048), stream=stream0)
        del buf599
        del buf604
        del buf609
        del buf613
        del div_61
        del mul_1
        del primals_4
        buf618 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf617, buf618, 768, 16, grid=grid(768), stream=stream0)
        del buf617
        buf620 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_3.run(buf619, buf620, 768, 16, grid=grid(768), stream=stream0)
        del buf619
        buf621 = empty((512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_19.run(buf621, 393216, grid=grid(393216), stream=stream0)
        buf622 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]
        triton_poi_fused_embedding_dense_backward_sum_20.run(slice_4, buf616, buf622, 393216, grid=grid(393216), stream=stream0)
        del buf616
        aten.index_put_(buf621, [slice_4], buf622, True)
        del buf622
        del slice_4
        buf625 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_21.run(buf625, 1536, grid=grid(1536), stream=stream0)
        aten.index_put_(buf625, [expand], buf626, True)
        del buf626
        del expand
        buf629 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_22.run(buf629, 23440896, grid=grid(23440896), stream=stream0)
        aten.index_put_(buf629, [primals_206], buf630, True)
        del buf630
        del primals_206
        return (buf629, buf625, buf621, buf618, buf620, reinterpret_tensor(buf610, (768, 768), (768, 1), 0), reinterpret_tensor(buf612, (768, ), (1, ), 0), reinterpret_tensor(buf605, (768, 768), (768, 1), 0), reinterpret_tensor(buf607, (768, ), (1, ), 0), reinterpret_tensor(buf600, (768, 768), (768, 1), 0), reinterpret_tensor(buf602, (768, ), (1, ), 0), reinterpret_tensor(buf588, (768, 768), (768, 1), 0), reinterpret_tensor(buf590, (768, ), (1, ), 0), buf584, buf586, reinterpret_tensor(buf577, (3072, 768), (768, 1), 0), reinterpret_tensor(buf579, (3072, ), (1, ), 0), reinterpret_tensor(buf572, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf574, (768, ), (1, ), 0), buf568, buf570, reinterpret_tensor(buf560, (768, 768), (768, 1), 0), reinterpret_tensor(buf562, (768, ), (1, ), 0), reinterpret_tensor(buf555, (768, 768), (768, 1), 0), reinterpret_tensor(buf557, (768, ), (1, ), 0), reinterpret_tensor(buf550, (768, 768), (768, 1), 0), reinterpret_tensor(buf552, (768, ), (1, ), 0), reinterpret_tensor(buf538, (768, 768), (768, 1), 0), reinterpret_tensor(buf540, (768, ), (1, ), 0), buf534, buf536, reinterpret_tensor(buf527, (3072, 768), (768, 1), 0), reinterpret_tensor(buf529, (3072, ), (1, ), 0), reinterpret_tensor(buf522, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf524, (768, ), (1, ), 0), buf518, buf520, reinterpret_tensor(buf510, (768, 768), (768, 1), 0), reinterpret_tensor(buf512, (768, ), (1, ), 0), reinterpret_tensor(buf505, (768, 768), (768, 1), 0), reinterpret_tensor(buf507, (768, ), (1, ), 0), reinterpret_tensor(buf500, (768, 768), (768, 1), 0), reinterpret_tensor(buf502, (768, ), (1, ), 0), reinterpret_tensor(buf488, (768, 768), (768, 1), 0), reinterpret_tensor(buf490, (768, ), (1, ), 0), buf484, buf486, reinterpret_tensor(buf477, (3072, 768), (768, 1), 0), reinterpret_tensor(buf479, (3072, ), (1, ), 0), reinterpret_tensor(buf472, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), buf468, buf470, reinterpret_tensor(buf460, (768, 768), (768, 1), 0), reinterpret_tensor(buf462, (768, ), (1, ), 0), reinterpret_tensor(buf455, (768, 768), (768, 1), 0), reinterpret_tensor(buf457, (768, ), (1, ), 0), reinterpret_tensor(buf450, (768, 768), (768, 1), 0), reinterpret_tensor(buf452, (768, ), (1, ), 0), reinterpret_tensor(buf438, (768, 768), (768, 1), 0), reinterpret_tensor(buf440, (768, ), (1, ), 0), buf434, buf436, reinterpret_tensor(buf427, (3072, 768), (768, 1), 0), reinterpret_tensor(buf429, (3072, ), (1, ), 0), reinterpret_tensor(buf422, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf424, (768, ), (1, ), 0), buf418, buf420, reinterpret_tensor(buf410, (768, 768), (768, 1), 0), reinterpret_tensor(buf412, (768, ), (1, ), 0), reinterpret_tensor(buf405, (768, 768), (768, 1), 0), reinterpret_tensor(buf407, (768, ), (1, ), 0), reinterpret_tensor(buf400, (768, 768), (768, 1), 0), reinterpret_tensor(buf402, (768, ), (1, ), 0), reinterpret_tensor(buf388, (768, 768), (768, 1), 0), reinterpret_tensor(buf390, (768, ), (1, ), 0), buf384, buf386, reinterpret_tensor(buf377, (3072, 768), (768, 1), 0), reinterpret_tensor(buf379, (3072, ), (1, ), 0), reinterpret_tensor(buf372, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf374, (768, ), (1, ), 0), buf368, buf370, reinterpret_tensor(buf360, (768, 768), (768, 1), 0), reinterpret_tensor(buf362, (768, ), (1, ), 0), reinterpret_tensor(buf355, (768, 768), (768, 1), 0), reinterpret_tensor(buf357, (768, ), (1, ), 0), reinterpret_tensor(buf350, (768, 768), (768, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), reinterpret_tensor(buf338, (768, 768), (768, 1), 0), reinterpret_tensor(buf340, (768, ), (1, ), 0), buf334, buf336, reinterpret_tensor(buf327, (3072, 768), (768, 1), 0), reinterpret_tensor(buf329, (3072, ), (1, ), 0), reinterpret_tensor(buf322, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf324, (768, ), (1, ), 0), buf318, buf320, reinterpret_tensor(buf310, (768, 768), (768, 1), 0), reinterpret_tensor(buf312, (768, ), (1, ), 0), reinterpret_tensor(buf305, (768, 768), (768, 1), 0), reinterpret_tensor(buf307, (768, ), (1, ), 0), reinterpret_tensor(buf300, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), reinterpret_tensor(buf288, (768, 768), (768, 1), 0), reinterpret_tensor(buf290, (768, ), (1, ), 0), buf284, buf286, reinterpret_tensor(buf277, (3072, 768), (768, 1), 0), reinterpret_tensor(buf279, (3072, ), (1, ), 0), reinterpret_tensor(buf272, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf274, (768, ), (1, ), 0), buf268, buf270, reinterpret_tensor(buf260, (768, 768), (768, 1), 0), reinterpret_tensor(buf262, (768, ), (1, ), 0), reinterpret_tensor(buf255, (768, 768), (768, 1), 0), reinterpret_tensor(buf257, (768, ), (1, ), 0), reinterpret_tensor(buf250, (768, 768), (768, 1), 0), reinterpret_tensor(buf252, (768, ), (1, ), 0), reinterpret_tensor(buf238, (768, 768), (768, 1), 0), reinterpret_tensor(buf240, (768, ), (1, ), 0), buf234, buf236, reinterpret_tensor(buf227, (3072, 768), (768, 1), 0), reinterpret_tensor(buf229, (3072, ), (1, ), 0), reinterpret_tensor(buf222, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf224, (768, ), (1, ), 0), buf218, buf220, reinterpret_tensor(buf210, (768, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf205, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf200, (768, 768), (768, 1), 0), reinterpret_tensor(buf202, (768, ), (1, ), 0), reinterpret_tensor(buf188, (768, 768), (768, 1), 0), reinterpret_tensor(buf190, (768, ), (1, ), 0), buf184, buf186, reinterpret_tensor(buf177, (3072, 768), (768, 1), 0), reinterpret_tensor(buf179, (3072, ), (1, ), 0), reinterpret_tensor(buf172, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf174, (768, ), (1, ), 0), buf168, buf170, reinterpret_tensor(buf160, (768, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), reinterpret_tensor(buf155, (768, 768), (768, 1), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), reinterpret_tensor(buf150, (768, 768), (768, 1), 0), reinterpret_tensor(buf152, (768, ), (1, ), 0), reinterpret_tensor(buf138, (768, 768), (768, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), buf134, buf136, reinterpret_tensor(buf127, (3072, 768), (768, 1), 0), reinterpret_tensor(buf129, (3072, ), (1, ), 0), reinterpret_tensor(buf122, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf124, (768, ), (1, ), 0), buf118, buf120, reinterpret_tensor(buf110, (768, 768), (768, 1), 0), reinterpret_tensor(buf112, (768, ), (1, ), 0), reinterpret_tensor(buf105, (768, 768), (768, 1), 0), reinterpret_tensor(buf107, (768, ), (1, ), 0), reinterpret_tensor(buf100, (768, 768), (768, 1), 0), reinterpret_tensor(buf102, (768, ), (1, ), 0), reinterpret_tensor(buf88, (768, 768), (768, 1), 0), reinterpret_tensor(buf90, (768, ), (1, ), 0), buf84, buf86, reinterpret_tensor(buf77, (3072, 768), (768, 1), 0), reinterpret_tensor(buf79, (3072, ), (1, ), 0), reinterpret_tensor(buf72, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf74, (768, ), (1, ), 0), buf68, buf70, reinterpret_tensor(buf60, (768, 768), (768, 1), 0), reinterpret_tensor(buf62, (768, ), (1, ), 0), reinterpret_tensor(buf55, (768, 768), (768, 1), 0), reinterpret_tensor(buf57, (768, ), (1, ), 0), reinterpret_tensor(buf50, (768, 768), (768, 1), 0), reinterpret_tensor(buf52, (768, ), (1, ), 0), reinterpret_tensor(buf38, (768, 768), (768, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), buf34, buf36, reinterpret_tensor(buf27, (3072, 768), (768, 1), 0), reinterpret_tensor(buf29, (3072, ), (1, ), 0), reinterpret_tensor(buf22, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf24, (768, ), (1, ), 0), buf18, buf20, reinterpret_tensor(buf11, (768, 768), (768, 1), 0), reinterpret_tensor(buf13, (768, ), (1, ), 0), buf6, buf8, reinterpret_tensor(buf1, (30522, 768), (768, 1), 0), reinterpret_tensor(buf2, (30522, ), (1, ), 0), None, None, None, )


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
    primals_206 = rand_strided((4, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    expand = rand_strided((4, 512), (0, 1), device='cuda:0', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_17 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_22 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_38 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_59 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_71 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((2048, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_72 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((4, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_353 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_386 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_454 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((48, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_519 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((48, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((48, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((4, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4, 512, 30522), (15627264, 30522, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, slice_4, mul_1, view, view_16, mul_3, view_18, addmm_4, view_20, mul_8, view_22, view_38, mul_10, view_40, addmm_10, view_42, mul_15, view_44, view_60, mul_17, view_62, addmm_16, view_64, mul_22, view_66, view_82, mul_24, view_84, addmm_22, view_86, mul_29, view_88, view_104, mul_31, view_106, addmm_28, view_108, mul_36, view_110, view_126, mul_38, view_128, addmm_34, view_130, mul_43, view_132, view_148, mul_45, view_150, addmm_40, view_152, mul_50, view_154, view_170, mul_52, view_172, addmm_46, view_174, mul_57, view_176, view_192, mul_59, view_194, addmm_52, view_196, mul_64, view_198, view_214, mul_66, view_216, addmm_58, view_218, mul_71, view_220, view_236, mul_73, view_238, addmm_64, view_240, mul_78, view_242, view_258, mul_80, view_260, addmm_70, view_262, mul_85, view_264, addmm_72, mul_90, view_266, permute_134, div_24, permute_138, div_25, permute_142, permute_146, div_26, permute_150, permute_155, permute_156, alias_12, permute_157, permute_158, permute_162, permute_167, permute_171, div_28, permute_175, permute_179, div_29, permute_183, permute_188, permute_189, alias_13, permute_190, permute_191, permute_195, permute_200, permute_204, div_31, permute_208, permute_212, div_32, permute_216, permute_221, permute_222, alias_14, permute_223, permute_224, permute_228, permute_233, permute_237, div_34, permute_241, permute_245, div_35, permute_249, permute_254, permute_255, alias_15, permute_256, permute_257, permute_261, permute_266, permute_270, div_37, permute_274, permute_278, div_38, permute_282, permute_287, permute_288, alias_16, permute_289, permute_290, permute_294, permute_299, permute_303, div_40, permute_307, permute_311, div_41, permute_315, permute_320, permute_321, alias_17, permute_322, permute_323, permute_327, permute_332, permute_336, div_43, permute_340, permute_344, div_44, permute_348, permute_353, permute_354, alias_18, permute_355, permute_356, permute_360, permute_365, permute_369, div_46, permute_373, permute_377, div_47, permute_381, permute_386, permute_387, alias_19, permute_388, permute_389, permute_393, permute_398, permute_402, div_49, permute_406, permute_410, div_50, permute_414, permute_419, permute_420, alias_20, permute_421, permute_422, permute_426, permute_431, permute_435, div_52, permute_439, permute_443, div_53, permute_447, permute_452, permute_453, alias_21, permute_454, permute_455, permute_459, permute_464, permute_468, div_55, permute_472, permute_476, div_56, permute_480, permute_485, permute_486, alias_22, permute_487, permute_488, permute_492, permute_497, permute_501, div_58, permute_505, permute_509, div_59, permute_513, permute_518, permute_519, alias_23, permute_520, permute_521, permute_525, permute_530, permute_534, div_61, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Bert', benchmark_compiled_module)
