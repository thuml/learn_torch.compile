
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


# kernel path: /tmp/torchinductor_youkaichao/jo/cjob3w7pnudbsmsobydrszwr3dj2myu5fipmnvxysf42srempntz.py
# Source Nodes: [x_12, x_20, x_28, x_36], Original ATen: [aten.add]
# x_12 => add_18
# x_20 => add_26
# x_28 => add_34
# x_36 => add_42
triton_poi_fused_add_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbmd7yhxuae3k2bc2delyhyjl6h2sewbslqgywgdldh5jv2iiuo.py
# Source Nodes: [x_101, x_107, x_71, x_77, x_83, x_89, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_101 => add_111
# x_107 => add_117
# x_71 => add_78
# x_77 => add_85
# x_83 => add_91
# x_89 => add_98
# x_95 => add_104
triton_poi_fused_add_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp9 = tl.load(in_ptr5 + (x0), None)
    tmp11 = tl.load(in_ptr6 + (x0), None)
    tmp13 = tl.load(in_ptr7 + (x0), None)
    tmp15 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 - tmp15
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3njwufgwb7ljmk42pbxetfowgtjedczuas2ld2bpd47mt3ssvv.py
# Source Nodes: [x_124, x_130, x_136, x_142, x_148, x_154, x_160, x_167], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_124 => add_136
# x_130 => add_143
# x_136 => add_149
# x_142 => add_156
# x_148 => add_162
# x_154 => add_169
# x_160 => add_175
# x_167 => add_182
triton_poi_fused_add_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp9 = tl.load(in_ptr5 + (x0), None)
    tmp11 = tl.load(in_ptr6 + (x0), None)
    tmp13 = tl.load(in_ptr7 + (x0), None)
    tmp15 = tl.load(in_ptr8 + (x0), None)
    tmp17 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr10 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 - tmp17
    tmp20 = tmp14 - tmp19
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp18, None)
    tl.store(out_ptr2 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxcoziy3tskt5fjlnvnatpj2gbgoaetln6w7qpegla3ygqg54ucr.py
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cn/ccngpyjdbjbiwstgql3xdok3quh2bvj5wbahwxatt64qvj7inj7w.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]

triton_red_fused_div_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 49)
        r1 = rindex % 49
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjrsfj4vdwsvcmfauvynu2bib6if2cr3i6vhemcfpzxnwfbavj3.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]

triton_poi_fused_div_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_native_batch_norm_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = (xindex // 49)
    x4 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x4), None)
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp2 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x4), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthj3eu4nsdejq5u5gnuzom7mffunkomr6dw6lkudhjnnni3ewza.py
# Source Nodes: [x_162], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
# x_162 => add_181, erf_21, mul_261
triton_poi_fused_convolution_backward_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
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


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqzma52oysqq7tpcito5gqeetprpqern2svqwjawfsurznszfji.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp0 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6w6slbdkpzsfubimnp6eatrsj3ugra4hsld5k3t5dkzya7xrksz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp4 = 0.002551020408163265
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp1 - tmp9
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 - tmp12
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmbcsavgp73igeusxaamvr2tgblbpheih6wjxcekp2px2jghbdd.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2352
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.08838834764831845
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvzlqvvvi3wbol6dnx4ddr2qschujrhxeyuxljk6fdnppqixccl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2304
    y1 = (yindex // 2304)
    x2 = xindex
    y3 = yindex
    tmp0 = y1 + (8*(y0 // 768))
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((128*x2) + (6272*((y0 // 128) % 6)) + (37632*y1) + (301056*(y0 // 768)) + (y0 % 128)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-301056) + x2 + (49*(y0 % 768)) + (37632*y1) + (301056*(y0 // 768))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-602112) + (128*x2) + (6272*((y0 // 128) % 6)) + (37632*y1) + (301056*(y0 // 768)) + (y0 % 128)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jfrb74tl3cnnt6fxn7o4t47g37lhpgmkp4wpnqmcif5y7rxqmx.py
# Source Nodes: [x_148, x_154], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_148 => add_162
# x_154 => add_169
triton_per_fused_add_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 - tmp10
    tmp12 = tmp0 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4rdwwaaoiqr4uf4xigmy6xoemmk4k3yzgzt2ymoc3biwu7i4bt.py
# Source Nodes: [x_148, x_154], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_148 => add_162
# x_154 => add_169
triton_poi_fused_add_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr1 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp0 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 + tmp21
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5ftbsfsc6n7wlh6xiesd3c2gcjt5isejfyf3eni2f6e5d3ojdc.py
# Source Nodes: [x_148], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_148 => add_162
triton_per_fused_add_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 - tmp8
    tmp10 = tmp0 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqkmiysbsqldczzknzh6tpp6uita6ghuxs4bngyt5on7v3jdjsi.py
# Source Nodes: [x_148], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_148 => add_162
triton_poi_fused_add_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp1 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsx5leiqz3h3bgzhzby5uquhxocnt55zscud47es5ggxlh6j4q5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwa3nuzhpaqsq6r2wi3azjhweseiv7jp33xabc32a6aucdqus6d.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.002551020408163265
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32ioya5flwwzayjm6awkgiw66d2ijisczjakathtrhxweqfor7l.py
# Source Nodes: [x_124, x_130, x_136], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_124 => add_136
# x_130 => add_143
# x_136 => add_149
triton_poi_fused_add_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3crox7uknnqhqvsygbijy7ztu4hwl57agyb6u5fvutfepfx5rp4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp4 = 0.002551020408163265
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp1 - tmp9
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 - tmp12
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpk36a7y53egy4kjehy3cprqtectove4ac5phry4i3axlcrgteo.py
# Source Nodes: [x_124], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_124 => add_136
triton_poi_fused_add_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 0.002551020408163265
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp1 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf3c7o6qxgc6od7rtrvtfabvoi6nptutzp7bqpthfvsz7eatximk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.002551020408163265
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mrlozj6bvdd3el4d3j2kpk3soconchtbz2osm34m7kheyrczhp.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (37632*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q2kse7ajuqvpecuigk4ms32ol673cmqnjkquo4hvxzbsnq2r4v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4fevnfsnlttwr3iy33tfuy4qmohempkcc3ayh5vow7dnpn5ugb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 768
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (37632*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaruid3fjp3gsxp6t6fbiigup323qv3bifu3sccplvxmxesjowj.py
# Source Nodes: [x_109], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
# x_109 => add_123, erf_17, mul_182
triton_poi_fused_convolution_backward_gelu_gelu_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_gelu_gelu_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsku7jrtamn2dsquqwedrkefr3rrmukltbwcu2wnl6eo2fjicyb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr2 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphfqn5n3loze2bhdgpaucsaocj7ickgvqzn7bjehbe4vxbbamu6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp4 = 0.0006377551020408163
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp1 - tmp9
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 - tmp12
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgccrwfc7vfghksj33ak2rzwmr4ltmvrprjqgi33vepeaz5a4il.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
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
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce62suhqg6uf5y4ae6apqdficngpjnt2yjbxr6s456ohgq7nkagm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1152
    y1 = (yindex // 1152)
    x2 = xindex
    y3 = yindex
    tmp0 = y1 + (8*(y0 // 384))
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x2) + (12544*((y0 // 64) % 6)) + (75264*y1) + (602112*(y0 // 384)) + (y0 % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-602112) + x2 + (196*(y0 % 384)) + (75264*y1) + (602112*(y0 // 384))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-1204224) + (64*x2) + (12544*((y0 // 64) % 6)) + (75264*y1) + (602112*(y0 // 384)) + (y0 % 64)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpydmjf6zppbzu4tltw2p6wzyrrp6aomzvjmckxyyllqf3uk6ywb.py
# Source Nodes: [x_101, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_101 => add_111
# x_95 => add_104
triton_red_fused_add_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bo/cbomuaz4keub2uyilo4fdksepn54ob5dg3zmdad445p4fkgpfdmw.py
# Source Nodes: [x_101, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_101 => add_111
# x_95 => add_104
triton_poi_fused_add_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr1 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp0 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 + tmp21
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mbobqcfvhb56n6pvk4lk2gqbvidywxrpgowkx265vmrswd3o7n.py
# Source Nodes: [x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_95 => add_104
triton_red_fused_add_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 - tmp7
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coafl6y6sg7wbsreyntjxls27zpvdxkr36tc5347zamr76itqwcg.py
# Source Nodes: [x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_95 => add_104
triton_poi_fused_add_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 0.0006377551020408163
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp1 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x4/cx4fnb3mjmh3bl4smadiqt2af23jv5itg3jtfbfoa4h55bs5pisf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn37e7aduxf23jhirfpqevtrylakkwobegs66dxryc5lviguwjar.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.0006377551020408163
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42gncjqpdlzwa7anwpd7perxqmdrbyltv6ab2iwpjhx6dqqogin.py
# Source Nodes: [x_71, x_77, x_83], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_71 => add_78
# x_77 => add_85
# x_83 => add_91
triton_poi_fused_add_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpfomvoc7c36s5n5h4zylbz5g3pa32qqri52b5aokq7wxd7jgw3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.0006377551020408163
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofvl7n7uzrk2mno2pbyp5xtgrhmbgjcvh45hbquwq7lxd32tvme.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75264*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvw55fh5xddw5fhc6txvy5ox7kg5awrgd2tyesnbu324qomen6eg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4vtpijlj2jvqhzcfdbjyzsqwuetcrvjvmah4oe4tbextudmppi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32urmg3a5kwa2zyu6o2bwdzckorbymef72l6ar72nyjtaumjipb.py
# Source Nodes: [x_44, x_52], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_44 => add_50
# x_52 => add_58
triton_red_fused_add_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhhzbatgxo52fbss3aqly2cd2hd7oepi272kp4xqtlyazrf6cgk.py
# Source Nodes: [x_44, x_52], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_44 => add_50
# x_52 => add_58
triton_poi_fused_add_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr1 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00015943877551020407
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp0 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22 + tmp21
    tl.store(in_out_ptr1 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/cia3ekvzx2kvt66ic555ykosz7cmgvhna2u7fxctqr76zcnzqn2v.py
# Source Nodes: [x_44], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_44 => add_50
triton_red_fused_add_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 - tmp7
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3kqqhxboaalkboyfypifr55qsqqtynpfubycvpekf36wmgwkmv.py
# Source Nodes: [x_44], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_44 => add_50
triton_poi_fused_add_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 0.00015943877551020407
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp1 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q4pnljsqrsaf7cyzqgb44rd67fjin7km2hha4i57u4j52gbxds.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4s/c4szrcwkt3atxrjuabopvauwwugznedoma4zftigt6hndvjey3ku.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.00015943877551020407
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshvurxxzmgk6ktqs3etlsfce7vzembl6xdebbvylu4vsjkjtf3l.py
# Source Nodes: [x_12, x_20, x_28], Original ATen: [aten.add, aten.native_batch_norm_backward]
# x_12 => add_18
# x_20 => add_26
# x_28 => add_34
triton_poi_fused_add_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x3), None)
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4fzp2g2fmczioeyqlxtupvl3hccdjvhwvpzlrarfbyyp6tkjgx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr2 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cvex2ytnplswio3w6ztlrnm77uzrdiqk73f2nw4mzwvnwevv2muf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp4 = 0.00015943877551020407
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp1 - tmp9
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 - tmp12
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr6dms4nchrkgnbjxnwxaspoarhurkcbyrcraubqzemvsw47v2v.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 - tmp3
    tmp6 = 0.00015943877551020407
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp1 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdaeyaa63xcdubwd2ymqi6cnpnpv7ya6vuzka3huv2le3q5koz2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[262144, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (150528*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cct77mkma2tck32jk6qgeditca2gxwhn2ejtkxlnp2yxzlymhtng.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00015943877551020407
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckknfwsycipqfrtbcawxfjdf2giply5xzrmtqu24oc4m54nltvbm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffivgya5iud4e7vulf63xwaxvcomx7jzecdtj77zsu6i4coeira.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdkw2ye5p6flg77wzwwze5ozvcogh4zuwavejkre6fffcpm3kga.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/wo/cwoih4aubf3tf6u56nhfdtycpbruqz6vykechyhbiku255mwtvhs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfm3eddqc24nwbies5nglyjkwdjbwxu4aub2kbvikodnalg5xgft.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, convolution, squeeze_1, relu, convolution_1, squeeze_4, clone, squeeze_7, add_15, convolution_2, clone_1, convolution_3, mul_26, convolution_4, squeeze_10, add_23, convolution_5, clone_3, convolution_6, mul_39, convolution_7, squeeze_13, add_31, convolution_8, clone_5, convolution_9, mul_52, convolution_10, squeeze_16, add_39, convolution_11, clone_7, convolution_12, mul_65, convolution_13, squeeze_19, add_47, convolution_14, clone_9, convolution_15, mul_78, convolution_16, squeeze_22, add_55, convolution_17, clone_11, convolution_18, mul_91, convolution_19, squeeze_25, add_63, convolution_20, clone_13, convolution_21, mul_104, add_66, convolution_23, squeeze_28, clone_15, squeeze_31, add_77, view_7, convolution_25, squeeze_34, add_83, convolution_26, clone_22, convolution_27, squeeze_37, add_90, view_15, convolution_29, squeeze_40, add_96, convolution_30, clone_30, convolution_31, squeeze_43, add_103, view_23, convolution_33, squeeze_46, add_109, convolution_34, clone_38, convolution_35, squeeze_49, add_116, view_31, convolution_37, squeeze_52, add_122, convolution_38, clone_46, add_124, convolution_40, squeeze_55, clone_48, squeeze_58, add_135, view_39, convolution_42, squeeze_61, add_141, convolution_43, clone_55, convolution_44, squeeze_64, add_148, view_47, convolution_46, squeeze_67, add_154, convolution_47, clone_63, convolution_48, squeeze_70, add_161, view_55, convolution_50, squeeze_73, add_167, convolution_51, clone_71, convolution_52, squeeze_76, add_174, view_63, convolution_54, squeeze_79, add_180, convolution_55, clone_79, convolution_56, squeeze_82, clone_81, permute_25, unsqueeze_114, unsqueeze_126, permute_30, permute_31, alias_9, permute_32, permute_33, unsqueeze_138, unsqueeze_150, permute_37, permute_38, alias_10, permute_39, permute_40, unsqueeze_162, unsqueeze_174, permute_44, permute_45, alias_11, permute_46, permute_47, unsqueeze_186, unsqueeze_198, permute_51, permute_52, alias_12, permute_53, permute_54, unsqueeze_210, unsqueeze_222, unsqueeze_234, permute_58, permute_59, alias_13, permute_60, permute_61, unsqueeze_246, unsqueeze_258, permute_65, permute_66, alias_14, permute_67, permute_68, unsqueeze_270, unsqueeze_282, permute_72, permute_73, alias_15, permute_74, permute_75, unsqueeze_294, unsqueeze_306, permute_79, permute_80, alias_16, permute_81, permute_82, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (32, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (192, 32, 4, 4), (512, 16, 4, 1))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_13, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_14, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_15, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_18, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_19, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_20, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_23, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_24, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_25, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_28, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_29, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_30, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_34, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_35, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_38, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_39, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_40, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_43, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (384, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_45, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_46, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_52, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_53, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_56, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_57, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_60, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_61, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_65, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_68, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_69, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_72, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_76, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_77, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_80, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_81, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_82, (768, 384, 2, 2), (1536, 4, 2, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_88, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_92, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_93, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_96, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_101, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_109, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_116, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_117, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_206, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(convolution_1, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_4, (192, ), (1, ))
    assert_size_stride(clone, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_7, (192, ), (1, ))
    assert_size_stride(add_15, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_2, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_1, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_3, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_26, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_4, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_10, (192, ), (1, ))
    assert_size_stride(add_23, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_5, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_3, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_6, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_39, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_7, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(add_31, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_8, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_5, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_9, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_52, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_10, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_16, (192, ), (1, ))
    assert_size_stride(add_39, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_11, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_7, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_12, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_65, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_19, (192, ), (1, ))
    assert_size_stride(add_47, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_14, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_9, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_15, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_78, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_16, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(add_55, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_17, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_11, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_18, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_91, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_19, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(add_63, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_20, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(clone_13, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(convolution_21, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(mul_104, (8, 384, 28, 28), (301056, 784, 28, 1))
    assert_size_stride(add_66, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(convolution_23, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_28, (384, ), (1, ))
    assert_size_stride(clone_15, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_31, (384, ), (1, ))
    assert_size_stride(add_77, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(view_7, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_34, (384, ), (1, ))
    assert_size_stride(add_83, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_26, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(clone_22, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_37, (384, ), (1, ))
    assert_size_stride(add_90, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(view_15, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_29, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_40, (384, ), (1, ))
    assert_size_stride(add_96, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_30, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(clone_30, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_43, (384, ), (1, ))
    assert_size_stride(add_103, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(view_23, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_46, (384, ), (1, ))
    assert_size_stride(add_109, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_34, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(clone_38, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(convolution_35, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_49, (384, ), (1, ))
    assert_size_stride(add_116, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(view_31, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(squeeze_52, (384, ), (1, ))
    assert_size_stride(add_122, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_38, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(clone_46, (8, 1536, 14, 14), (301056, 196, 14, 1))
    assert_size_stride(add_124, (8, 384, 14, 14), (75264, 196, 14, 1))
    assert_size_stride(convolution_40, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(clone_48, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_58, (768, ), (1, ))
    assert_size_stride(add_135, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(view_39, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_42, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_61, (768, ), (1, ))
    assert_size_stride(add_141, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_43, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(clone_55, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(convolution_44, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(add_148, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(view_47, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_46, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(add_154, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_47, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(clone_63, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(convolution_48, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_70, (768, ), (1, ))
    assert_size_stride(add_161, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(view_55, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_50, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(add_167, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_51, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(clone_71, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(convolution_52, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_76, (768, ), (1, ))
    assert_size_stride(add_174, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(view_63, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_54, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(add_180, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(convolution_55, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(clone_79, (8, 3072, 7, 7), (150528, 49, 7, 1))
    assert_size_stride(convolution_56, (8, 768, 7, 7), (37632, 49, 7, 1))
    assert_size_stride(squeeze_82, (768, ), (1, ))
    assert_size_stride(clone_81, (8, 768), (768, 1))
    assert_size_stride(permute_25, (1000, 768), (768, 1))
    assert_size_stride(unsqueeze_114, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_126, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_30, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_31, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_9, (8, 6, 49, 49), (14406, 2401, 49, 1))
    assert_size_stride(permute_32, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_33, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_138, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_150, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_37, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_38, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_10, (8, 6, 49, 49), (14406, 2401, 49, 1))
    assert_size_stride(permute_39, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_40, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_162, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_44, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_45, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_11, (8, 6, 49, 49), (14406, 2401, 49, 1))
    assert_size_stride(permute_46, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_47, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_186, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_198, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_51, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_52, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_12, (8, 6, 49, 49), (14406, 2401, 49, 1))
    assert_size_stride(permute_53, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_54, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_210, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_58, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_59, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_13, (8, 6, 196, 196), (230496, 38416, 196, 1))
    assert_size_stride(permute_60, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_61, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_246, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_65, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_66, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_14, (8, 6, 196, 196), (230496, 38416, 196, 1))
    assert_size_stride(permute_67, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_68, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_270, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_72, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_73, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_15, (8, 6, 196, 196), (230496, 38416, 196, 1))
    assert_size_stride(permute_74, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_75, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_294, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_79, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_80, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_16, (8, 6, 196, 196), (230496, 38416, 196, 1))
    assert_size_stride(permute_81, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_82, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_318, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_20, x_28, x_36], Original ATen: [aten.add]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_0.run(clone, convolution_4, convolution_7, convolution_10, convolution_13, buf0, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_13
        buf1 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf144 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_107, x_71, x_77, x_83, x_89, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_1.run(clone_15, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35, convolution_37, unsqueeze_234, buf1, buf144, 602112, grid=grid(602112), stream=stream0)
        del convolution_31
        del convolution_37
        del unsqueeze_234
        buf2 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf7 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf19 = empty((8, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124, x_130, x_136, x_142, x_148, x_154, x_160, x_167], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_2.run(clone_48, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, unsqueeze_114, unsqueeze_126, buf2, buf7, buf19, 301056, grid=grid(301056), stream=stream0)
        del convolution_48
        del convolution_54
        del convolution_56
        del unsqueeze_114
        del unsqueeze_126
        buf3 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_25, out=buf3)
        del permute_25
        buf4 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_81, out=buf4)
        del clone_81
        buf5 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(tangents_1, buf5, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        buf10 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]
        triton_red_fused_div_native_batch_norm_backward_4.run(buf3, buf7, squeeze_82, buf6, buf8, buf10, 768, 392, grid=grid(768), stream=stream0)
        buf9 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward]
        triton_poi_fused_div_native_batch_norm_backward_5.run(buf9, buf3, buf8, squeeze_82, buf6, primals_118, 301056, grid=grid(301056), stream=stream0)
        del buf3
        del primals_118
        del squeeze_82
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf11 = aten.convolution_backward(buf9, clone_79, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_79
        del primals_117
        buf12 = buf11[0]
        buf13 = buf11[1]
        del buf11
        buf14 = buf12; del buf12  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_6.run(buf14, convolution_55, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_55
        # Source Nodes: [x_162], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf15 = aten.convolution_backward(buf14, add_180, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_180
        del buf14
        del primals_116
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = buf8; del buf8  # reuse
        buf20 = empty((768, ), device='cuda', dtype=torch.float32)
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf16, buf19, squeeze_79, buf18, buf20, buf21, 768, 392, grid=grid(768), stream=stream0)
        buf22 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_8.run(buf22, buf9, buf19, buf20, squeeze_79, buf18, primals_114, 301056, grid=grid(301056), stream=stream0)
        del primals_114
        del squeeze_79
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf23 = aten.convolution_backward(buf22, view_63, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_113
        del view_63
        buf24 = buf23[0]
        buf25 = buf23[1]
        del buf23
        buf26 = reinterpret_tensor(buf9, (48, 49, 128), (6272, 128, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_30, reinterpret_tensor(buf24, (48, 49, 128), (6272, 1, 49), 0), out=buf26)
        del permute_30
        buf27 = empty((48, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (48, 49, 128), (6272, 1, 49), 0), permute_31, out=buf27)
        del permute_31
        buf29 = empty((8, 6, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_9.run(buf27, alias_9, buf29, 2352, 49, grid=grid(2352), stream=stream0)
        del alias_9
        buf30 = reinterpret_tensor(buf24, (48, 128, 49), (6272, 49, 1), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_32, reinterpret_tensor(buf29, (48, 49, 49), (2401, 49, 1), 0), out=buf30)
        del permute_32
        buf31 = reinterpret_tensor(buf19, (48, 49, 128), (6272, 128, 1), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (48, 49, 49), (2401, 49, 1), 0), permute_33, out=buf31)
        del permute_33
        buf32 = empty((8, 2304, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_10.run(buf31, buf30, buf26, buf32, 18432, 49, grid=grid(18432, 49), stream=stream0)
        del buf26
        del buf30
        del buf31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf33 = aten.convolution_backward(buf32, add_174, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_174
        del primals_112
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf36 = buf20; del buf20  # reuse
        buf37 = empty((768, ), device='cuda', dtype=torch.float32)
        buf39 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148, x_154], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_11.run(buf34, buf2, convolution_50, convolution_52, unsqueeze_138, squeeze_76, buf36, buf37, buf39, 768, 392, grid=grid(768), stream=stream0)
        buf38 = buf34; del buf34  # reuse
        buf40 = buf22; del buf22  # reuse
        # Source Nodes: [x_148, x_154], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_12.run(buf38, buf40, buf2, convolution_50, convolution_52, unsqueeze_138, buf37, squeeze_76, buf36, primals_110, 301056, grid=grid(301056), stream=stream0)
        del convolution_52
        del primals_110
        del squeeze_76
        del unsqueeze_138
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf41 = aten.convolution_backward(buf40, clone_71, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_71
        del primals_109
        buf42 = buf41[0]
        buf43 = buf41[1]
        del buf41
        buf44 = buf42; del buf42  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_6.run(buf44, convolution_51, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_51
        # Source Nodes: [x_150], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf45 = aten.convolution_backward(buf44, add_167, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_167
        del buf44
        del primals_108
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf48 = buf37; del buf37  # reuse
        buf49 = empty((768, ), device='cuda', dtype=torch.float32)
        buf50 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_148], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_13.run(buf46, buf2, convolution_50, unsqueeze_150, squeeze_73, buf48, buf49, buf50, 768, 392, grid=grid(768), stream=stream0)
        buf51 = buf40; del buf40  # reuse
        # Source Nodes: [x_148], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_14.run(buf51, buf46, buf2, convolution_50, unsqueeze_150, buf49, squeeze_73, buf48, primals_106, 301056, grid=grid(301056), stream=stream0)
        del convolution_50
        del primals_106
        del squeeze_73
        del unsqueeze_150
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf52 = aten.convolution_backward(buf51, view_55, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_105
        del view_55
        buf53 = buf52[0]
        buf54 = buf52[1]
        del buf52
        buf55 = reinterpret_tensor(buf46, (48, 49, 128), (6272, 128, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_37, reinterpret_tensor(buf53, (48, 49, 128), (6272, 1, 49), 0), out=buf55)
        del permute_37
        buf56 = reinterpret_tensor(buf29, (48, 49, 49), (2401, 49, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (48, 49, 128), (6272, 1, 49), 0), permute_38, out=buf56)
        del permute_38
        buf58 = reinterpret_tensor(buf27, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_9.run(buf56, alias_10, buf58, 2352, 49, grid=grid(2352), stream=stream0)
        del alias_10
        buf59 = reinterpret_tensor(buf53, (48, 128, 49), (6272, 49, 1), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_39, reinterpret_tensor(buf58, (48, 49, 49), (2401, 49, 1), 0), out=buf59)
        del permute_39
        buf60 = reinterpret_tensor(buf38, (48, 49, 128), (6272, 128, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (48, 49, 49), (2401, 49, 1), 0), permute_40, out=buf60)
        del permute_40
        buf61 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_10.run(buf60, buf59, buf55, buf61, 18432, 49, grid=grid(18432, 49), stream=stream0)
        del buf55
        del buf59
        del buf60
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf62 = aten.convolution_backward(buf61, add_161, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_161
        del primals_104
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        buf65 = buf49; del buf49  # reuse
        buf66 = empty((768, ), device='cuda', dtype=torch.float32)
        buf67 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf63, buf2, unsqueeze_162, squeeze_70, buf65, buf66, buf67, 768, 392, grid=grid(768), stream=stream0)
        buf68 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_16.run(buf68, buf51, buf63, unsqueeze_162, buf66, squeeze_70, buf65, primals_102, 301056, grid=grid(301056), stream=stream0)
        del buf51
        del primals_102
        del squeeze_70
        del unsqueeze_162
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = aten.convolution_backward(buf68, clone_63, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_63
        del primals_101
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = buf70; del buf70  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_6.run(buf72, convolution_47, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_47
        # Source Nodes: [x_138], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf73 = aten.convolution_backward(buf72, add_154, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_154
        del buf72
        del primals_100
        buf74 = buf73[0]
        buf75 = buf73[1]
        del buf73
        buf77 = buf63; del buf63  # reuse
        # Source Nodes: [x_124, x_130, x_136], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_17.run(clone_48, convolution_42, convolution_44, convolution_46, unsqueeze_174, buf77, 301056, grid=grid(301056), stream=stream0)
        del convolution_46
        del unsqueeze_174
        buf76 = buf66; del buf66  # reuse
        buf78 = empty((768, ), device='cuda', dtype=torch.float32)
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf74, buf77, squeeze_67, buf76, buf78, buf79, 768, 392, grid=grid(768), stream=stream0)
        buf80 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_18.run(buf80, buf74, buf77, buf78, squeeze_67, buf76, primals_98, 301056, grid=grid(301056), stream=stream0)
        del primals_98
        del squeeze_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf81 = aten.convolution_backward(buf80, view_47, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_97
        del view_47
        buf82 = buf81[0]
        buf83 = buf81[1]
        del buf81
        buf84 = reinterpret_tensor(buf77, (48, 49, 128), (6272, 128, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_44, reinterpret_tensor(buf82, (48, 49, 128), (6272, 1, 49), 0), out=buf84)
        del permute_44
        buf85 = reinterpret_tensor(buf58, (48, 49, 49), (2401, 49, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (48, 49, 128), (6272, 1, 49), 0), permute_45, out=buf85)
        del permute_45
        buf87 = reinterpret_tensor(buf56, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_9.run(buf85, alias_11, buf87, 2352, 49, grid=grid(2352), stream=stream0)
        del alias_11
        buf88 = reinterpret_tensor(buf82, (48, 128, 49), (6272, 49, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_46, reinterpret_tensor(buf87, (48, 49, 49), (2401, 49, 1), 0), out=buf88)
        del permute_46
        buf89 = reinterpret_tensor(buf74, (48, 49, 128), (6272, 128, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf87, (48, 49, 49), (2401, 49, 1), 0), permute_47, out=buf89)
        del permute_47
        buf90 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_10.run(buf89, buf88, buf84, buf90, 18432, 49, grid=grid(18432, 49), stream=stream0)
        del buf84
        del buf88
        del buf89
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf91 = aten.convolution_backward(buf90, add_148, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_148
        del primals_96
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        buf94 = buf78; del buf78  # reuse
        buf95 = empty((768, ), device='cuda', dtype=torch.float32)
        buf97 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124, x_130], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_11.run(buf92, clone_48, convolution_42, convolution_44, unsqueeze_186, squeeze_64, buf94, buf95, buf97, 768, 392, grid=grid(768), stream=stream0)
        buf96 = buf92; del buf92  # reuse
        buf98 = buf80; del buf80  # reuse
        # Source Nodes: [x_124, x_130], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_12.run(buf96, buf98, clone_48, convolution_42, convolution_44, unsqueeze_186, buf95, squeeze_64, buf94, primals_94, 301056, grid=grid(301056), stream=stream0)
        del convolution_44
        del primals_94
        del squeeze_64
        del unsqueeze_186
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf98, clone_55, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_55
        del primals_93
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = buf100; del buf100  # reuse
        # Source Nodes: [x_126], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_6.run(buf102, convolution_43, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_43
        # Source Nodes: [x_126], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf103 = aten.convolution_backward(buf102, add_141, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_141
        del buf102
        del primals_92
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = buf95; del buf95  # reuse
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        buf108 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_124], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_13.run(buf104, clone_48, convolution_42, unsqueeze_198, squeeze_61, buf106, buf107, buf108, 768, 392, grid=grid(768), stream=stream0)
        buf109 = buf104; del buf104  # reuse
        # Source Nodes: [x_124], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_19.run(buf109, buf98, clone_48, convolution_42, unsqueeze_198, buf107, squeeze_61, buf106, primals_90, 301056, grid=grid(301056), stream=stream0)
        del convolution_42
        del primals_90
        del squeeze_61
        del unsqueeze_198
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf110 = aten.convolution_backward(buf109, view_39, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_89
        del view_39
        buf111 = buf110[0]
        buf112 = buf110[1]
        del buf110
        buf113 = reinterpret_tensor(buf98, (48, 49, 128), (6272, 128, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_51, reinterpret_tensor(buf111, (48, 49, 128), (6272, 1, 49), 0), out=buf113)
        del permute_51
        buf114 = reinterpret_tensor(buf87, (48, 49, 49), (2401, 49, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (48, 49, 128), (6272, 1, 49), 0), permute_52, out=buf114)
        del permute_52
        buf116 = reinterpret_tensor(buf85, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_9.run(buf114, alias_12, buf116, 2352, 49, grid=grid(2352), stream=stream0)
        del alias_12
        del buf114
        buf117 = reinterpret_tensor(buf111, (48, 128, 49), (6272, 49, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_53, reinterpret_tensor(buf116, (48, 49, 49), (2401, 49, 1), 0), out=buf117)
        del permute_53
        buf118 = reinterpret_tensor(buf96, (48, 49, 128), (6272, 128, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (48, 49, 49), (2401, 49, 1), 0), permute_54, out=buf118)
        del buf116
        del permute_54
        buf119 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_10.run(buf118, buf117, buf113, buf119, 18432, 49, grid=grid(18432, 49), stream=stream0)
        del buf113
        del buf117
        del buf118
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf120 = aten.convolution_backward(buf119, add_135, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_135
        del buf119
        del primals_88
        buf121 = buf120[0]
        buf122 = buf120[1]
        del buf120
        buf123 = buf107; del buf107  # reuse
        buf124 = empty((768, ), device='cuda', dtype=torch.float32)
        buf125 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf121, clone_48, unsqueeze_210, squeeze_58, buf123, buf124, buf125, 768, 392, grid=grid(768), stream=stream0)
        buf126 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_20.run(buf126, buf121, clone_48, unsqueeze_210, buf124, squeeze_58, buf123, primals_86, 301056, grid=grid(301056), stream=stream0)
        del buf121
        del clone_48
        del primals_86
        del squeeze_58
        del unsqueeze_210
        buf127 = empty((1, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf126, buf127, 37632, 8, grid=grid(37632), stream=stream0)
        buf128 = buf124; del buf124  # reuse
        buf129 = empty((768, ), device='cuda', dtype=torch.float32)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_15.run(buf126, convolution_40, unsqueeze_222, squeeze_55, buf128, buf129, buf131, 768, 392, grid=grid(768), stream=stream0)
        buf130 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_22.run(buf130, convolution_40, unsqueeze_222, buf129, squeeze_55, buf128, primals_84, 301056, grid=grid(301056), stream=stream0)
        del convolution_40
        del primals_84
        del squeeze_55
        del unsqueeze_222
        buf132 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf130, buf132, 768, 392, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf133 = aten.convolution_backward(buf130, add_124, primals_82, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_124
        del buf130
        del primals_82
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf136 = aten.convolution_backward(buf134, clone_46, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_46
        del primals_81
        buf137 = buf136[0]
        buf138 = buf136[1]
        del buf136
        buf139 = buf137; del buf137  # reuse
        # Source Nodes: [x_109], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf139, convolution_38, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_38
        # Source Nodes: [x_109], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf140 = aten.convolution_backward(buf139, add_122, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_122
        del buf139
        del primals_80
        buf141 = buf140[0]
        buf142 = buf140[1]
        del buf140
        buf143 = empty((384, ), device='cuda', dtype=torch.float32)
        buf145 = empty((384, ), device='cuda', dtype=torch.float32)
        buf146 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_25.run(buf141, buf144, squeeze_52, buf143, buf145, buf146, 384, 1568, grid=grid(384), stream=stream0)
        buf147 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_26.run(buf147, buf141, buf144, buf145, squeeze_52, buf143, primals_78, 602112, grid=grid(602112), stream=stream0)
        del primals_78
        del squeeze_52
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf148 = aten.convolution_backward(buf147, view_31, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_77
        del view_31
        buf149 = buf148[0]
        buf150 = buf148[1]
        del buf148
        buf151 = reinterpret_tensor(buf144, (48, 196, 64), (12544, 64, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_58, reinterpret_tensor(buf149, (48, 196, 64), (12544, 1, 196), 0), out=buf151)
        del permute_58
        buf152 = empty((48, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (48, 196, 64), (12544, 1, 196), 0), permute_59, out=buf152)
        del permute_59
        buf154 = empty((8, 6, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_27.run(buf152, alias_13, buf154, 9408, 196, grid=grid(9408), stream=stream0)
        del alias_13
        buf155 = reinterpret_tensor(buf149, (48, 64, 196), (12544, 196, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_60, reinterpret_tensor(buf154, (48, 196, 196), (38416, 196, 1), 0), out=buf155)
        del permute_60
        buf156 = reinterpret_tensor(buf141, (48, 196, 64), (12544, 64, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (48, 196, 196), (38416, 196, 1), 0), permute_61, out=buf156)
        del permute_61
        buf157 = empty((8, 1152, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_28.run(buf156, buf155, buf151, buf157, 9216, 196, grid=grid(9216, 196), stream=stream0)
        del buf151
        del buf155
        del buf156
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf158 = aten.convolution_backward(buf157, add_116, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_116
        del primals_76
        buf159 = buf158[0]
        buf160 = buf158[1]
        del buf158
        buf161 = buf145; del buf145  # reuse
        buf162 = empty((384, ), device='cuda', dtype=torch.float32)
        buf164 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_29.run(buf159, buf1, convolution_33, convolution_35, unsqueeze_246, squeeze_49, buf161, buf162, buf164, 384, 1568, grid=grid(384), stream=stream0)
        buf163 = buf159; del buf159  # reuse
        buf165 = buf147; del buf147  # reuse
        # Source Nodes: [x_101, x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_30.run(buf163, buf165, buf1, convolution_33, convolution_35, unsqueeze_246, buf162, squeeze_49, buf161, primals_74, 602112, grid=grid(602112), stream=stream0)
        del convolution_35
        del primals_74
        del squeeze_49
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf166 = aten.convolution_backward(buf165, clone_38, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_38
        del primals_73
        buf167 = buf166[0]
        buf168 = buf166[1]
        del buf166
        buf169 = buf167; del buf167  # reuse
        # Source Nodes: [x_97], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf169, convolution_34, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_34
        # Source Nodes: [x_97], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf170 = aten.convolution_backward(buf169, add_109, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del buf169
        del primals_72
        buf171 = buf170[0]
        buf172 = buf170[1]
        del buf170
        buf173 = buf162; del buf162  # reuse
        buf174 = empty((384, ), device='cuda', dtype=torch.float32)
        buf175 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_31.run(buf171, buf1, convolution_33, unsqueeze_258, squeeze_46, buf173, buf174, buf175, 384, 1568, grid=grid(384), stream=stream0)
        buf176 = buf165; del buf165  # reuse
        # Source Nodes: [x_95], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_32.run(buf176, buf171, buf1, convolution_33, unsqueeze_258, buf174, squeeze_46, buf173, primals_70, 602112, grid=grid(602112), stream=stream0)
        del convolution_33
        del primals_70
        del squeeze_46
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf177 = aten.convolution_backward(buf176, view_23, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_69
        del view_23
        buf178 = buf177[0]
        buf179 = buf177[1]
        del buf177
        buf180 = reinterpret_tensor(buf171, (48, 196, 64), (12544, 64, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_65, reinterpret_tensor(buf178, (48, 196, 64), (12544, 1, 196), 0), out=buf180)
        del permute_65
        buf181 = reinterpret_tensor(buf154, (48, 196, 196), (38416, 196, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf178, (48, 196, 64), (12544, 1, 196), 0), permute_66, out=buf181)
        del permute_66
        buf183 = reinterpret_tensor(buf152, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_27.run(buf181, alias_14, buf183, 9408, 196, grid=grid(9408), stream=stream0)
        del alias_14
        buf184 = reinterpret_tensor(buf178, (48, 64, 196), (12544, 196, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_67, reinterpret_tensor(buf183, (48, 196, 196), (38416, 196, 1), 0), out=buf184)
        del permute_67
        buf185 = reinterpret_tensor(buf163, (48, 196, 64), (12544, 64, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (48, 196, 196), (38416, 196, 1), 0), permute_68, out=buf185)
        del permute_68
        buf186 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_28.run(buf185, buf184, buf180, buf186, 9216, 196, grid=grid(9216, 196), stream=stream0)
        del buf180
        del buf184
        del buf185
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf187 = aten.convolution_backward(buf186, add_103, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_103
        del primals_68
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf190 = buf174; del buf174  # reuse
        buf191 = empty((384, ), device='cuda', dtype=torch.float32)
        buf192 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf188, buf1, unsqueeze_270, squeeze_43, buf190, buf191, buf192, 384, 1568, grid=grid(384), stream=stream0)
        buf193 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_34.run(buf193, buf176, buf188, unsqueeze_270, buf191, squeeze_43, buf190, primals_66, 602112, grid=grid(602112), stream=stream0)
        del buf176
        del primals_66
        del squeeze_43
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf194 = aten.convolution_backward(buf193, clone_30, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_30
        del primals_65
        buf195 = buf194[0]
        buf196 = buf194[1]
        del buf194
        buf197 = buf195; del buf195  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf197, convolution_30, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_30
        # Source Nodes: [x_85], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf198 = aten.convolution_backward(buf197, add_96, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_96
        del buf197
        del primals_64
        buf199 = buf198[0]
        buf200 = buf198[1]
        del buf198
        buf202 = buf188; del buf188  # reuse
        # Source Nodes: [x_71, x_77, x_83], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_35.run(clone_15, convolution_25, convolution_27, convolution_29, unsqueeze_282, buf202, 602112, grid=grid(602112), stream=stream0)
        del convolution_29
        del unsqueeze_282
        buf201 = buf191; del buf191  # reuse
        buf203 = empty((384, ), device='cuda', dtype=torch.float32)
        buf204 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_25.run(buf199, buf202, squeeze_40, buf201, buf203, buf204, 384, 1568, grid=grid(384), stream=stream0)
        buf205 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_26.run(buf205, buf199, buf202, buf203, squeeze_40, buf201, primals_62, 602112, grid=grid(602112), stream=stream0)
        del primals_62
        del squeeze_40
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf206 = aten.convolution_backward(buf205, view_15, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_61
        del view_15
        buf207 = buf206[0]
        buf208 = buf206[1]
        del buf206
        buf209 = reinterpret_tensor(buf202, (48, 196, 64), (12544, 64, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_72, reinterpret_tensor(buf207, (48, 196, 64), (12544, 1, 196), 0), out=buf209)
        del permute_72
        buf210 = reinterpret_tensor(buf183, (48, 196, 196), (38416, 196, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (48, 196, 64), (12544, 1, 196), 0), permute_73, out=buf210)
        del permute_73
        buf212 = reinterpret_tensor(buf181, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_27.run(buf210, alias_15, buf212, 9408, 196, grid=grid(9408), stream=stream0)
        del alias_15
        buf213 = reinterpret_tensor(buf207, (48, 64, 196), (12544, 196, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_74, reinterpret_tensor(buf212, (48, 196, 196), (38416, 196, 1), 0), out=buf213)
        del permute_74
        buf214 = reinterpret_tensor(buf199, (48, 196, 64), (12544, 64, 1), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (48, 196, 196), (38416, 196, 1), 0), permute_75, out=buf214)
        del permute_75
        buf215 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_28.run(buf214, buf213, buf209, buf215, 9216, 196, grid=grid(9216, 196), stream=stream0)
        del buf209
        del buf213
        del buf214
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf216 = aten.convolution_backward(buf215, add_90, primals_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_90
        del primals_60
        buf217 = buf216[0]
        buf218 = buf216[1]
        del buf216
        buf219 = buf203; del buf203  # reuse
        buf220 = empty((384, ), device='cuda', dtype=torch.float32)
        buf222 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_77], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_29.run(buf217, clone_15, convolution_25, convolution_27, unsqueeze_294, squeeze_37, buf219, buf220, buf222, 384, 1568, grid=grid(384), stream=stream0)
        buf221 = buf217; del buf217  # reuse
        buf223 = buf205; del buf205  # reuse
        # Source Nodes: [x_71, x_77], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_30.run(buf221, buf223, clone_15, convolution_25, convolution_27, unsqueeze_294, buf220, squeeze_37, buf219, primals_58, 602112, grid=grid(602112), stream=stream0)
        del convolution_27
        del primals_58
        del squeeze_37
        del unsqueeze_294
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf224 = aten.convolution_backward(buf223, clone_22, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_22
        del primals_57
        buf225 = buf224[0]
        buf226 = buf224[1]
        del buf224
        buf227 = buf225; del buf225  # reuse
        # Source Nodes: [x_73], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf227, convolution_26, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_26
        # Source Nodes: [x_73], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf228 = aten.convolution_backward(buf227, add_83, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_83
        del buf227
        del primals_56
        buf229 = buf228[0]
        buf230 = buf228[1]
        del buf228
        buf231 = buf220; del buf220  # reuse
        buf232 = empty((384, ), device='cuda', dtype=torch.float32)
        buf233 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_31.run(buf229, clone_15, convolution_25, unsqueeze_306, squeeze_34, buf231, buf232, buf233, 384, 1568, grid=grid(384), stream=stream0)
        buf234 = buf223; del buf223  # reuse
        # Source Nodes: [x_71], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_32.run(buf234, buf229, clone_15, convolution_25, unsqueeze_306, buf232, squeeze_34, buf231, primals_54, 602112, grid=grid(602112), stream=stream0)
        del convolution_25
        del primals_54
        del squeeze_34
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf235 = aten.convolution_backward(buf234, view_7, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_53
        del view_7
        buf236 = buf235[0]
        buf237 = buf235[1]
        del buf235
        buf238 = reinterpret_tensor(buf229, (48, 196, 64), (12544, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_79, reinterpret_tensor(buf236, (48, 196, 64), (12544, 1, 196), 0), out=buf238)
        del permute_79
        buf239 = reinterpret_tensor(buf212, (48, 196, 196), (38416, 196, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (48, 196, 64), (12544, 1, 196), 0), permute_80, out=buf239)
        del permute_80
        buf241 = reinterpret_tensor(buf210, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_27.run(buf239, alias_16, buf241, 9408, 196, grid=grid(9408), stream=stream0)
        del alias_16
        del buf239
        buf242 = reinterpret_tensor(buf236, (48, 64, 196), (12544, 196, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_81, reinterpret_tensor(buf241, (48, 196, 196), (38416, 196, 1), 0), out=buf242)
        del permute_81
        buf243 = reinterpret_tensor(buf221, (48, 196, 64), (12544, 64, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (48, 196, 196), (38416, 196, 1), 0), permute_82, out=buf243)
        del buf241
        del permute_82
        buf244 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_28.run(buf243, buf242, buf238, buf244, 9216, 196, grid=grid(9216, 196), stream=stream0)
        del buf238
        del buf242
        del buf243
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf245 = aten.convolution_backward(buf244, add_77, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_77
        del buf244
        del primals_52
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = buf232; del buf232  # reuse
        buf249 = empty((384, ), device='cuda', dtype=torch.float32)
        buf250 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf246, clone_15, unsqueeze_318, squeeze_31, buf248, buf249, buf250, 384, 1568, grid=grid(384), stream=stream0)
        buf251 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_36.run(buf251, buf246, clone_15, unsqueeze_318, buf249, squeeze_31, buf248, primals_50, 602112, grid=grid(602112), stream=stream0)
        del buf246
        del clone_15
        del primals_50
        del squeeze_31
        del unsqueeze_318
        buf252 = empty((1, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf251, buf252, 75264, 8, grid=grid(75264), stream=stream0)
        buf253 = buf249; del buf249  # reuse
        buf254 = empty((384, ), device='cuda', dtype=torch.float32)
        buf256 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf251, convolution_23, unsqueeze_330, squeeze_28, buf253, buf254, buf256, 384, 1568, grid=grid(384), stream=stream0)
        buf255 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_38.run(buf255, convolution_23, unsqueeze_330, buf254, squeeze_28, buf253, primals_48, 602112, grid=grid(602112), stream=stream0)
        del convolution_23
        del primals_48
        del squeeze_28
        del unsqueeze_330
        buf257 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_39.run(buf255, buf257, 384, 1568, grid=grid(384), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf258 = aten.convolution_backward(buf255, add_66, primals_46, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_66
        del buf255
        del primals_46
        buf259 = buf258[0]
        buf260 = buf258[1]
        del buf258
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf261 = aten.convolution_backward(buf259, mul_104, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_104
        del primals_45
        buf262 = buf261[0]
        buf263 = buf261[1]
        del buf261
        buf264 = buf262; del buf262  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf264, convolution_21, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_21
        # Source Nodes: [x_57], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf265 = aten.convolution_backward(buf264, clone_13, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf264
        del clone_13
        del primals_44
        buf266 = buf265[0]
        buf267 = buf265[1]
        del buf265
        buf268 = buf266; del buf266  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf268, convolution_20, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_20
        # Source Nodes: [x_54], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf269 = aten.convolution_backward(buf268, add_63, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_63
        del buf268
        del primals_43
        buf270 = buf269[0]
        buf271 = buf269[1]
        del buf269
        buf272 = empty((192, ), device='cuda', dtype=torch.float32)
        buf273 = empty((192, ), device='cuda', dtype=torch.float32)
        buf275 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_52], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_40.run(buf270, buf0, convolution_16, convolution_19, unsqueeze_342, squeeze_25, buf272, buf273, buf275, 192, 6272, grid=grid(192), stream=stream0)
        buf274 = buf270; del buf270  # reuse
        buf276 = buf259; del buf259  # reuse
        # Source Nodes: [x_44, x_52], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_41.run(buf274, buf276, buf0, convolution_16, convolution_19, unsqueeze_342, buf273, squeeze_25, buf272, primals_41, 1204224, grid=grid(1204224), stream=stream0)
        del buf274
        del convolution_19
        del primals_41
        del squeeze_25
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf277 = aten.convolution_backward(buf276, mul_91, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_91
        del primals_40
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf278; del buf278  # reuse
        # Source Nodes: [x_49], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf280, convolution_18, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_18
        # Source Nodes: [x_49], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf281 = aten.convolution_backward(buf280, clone_11, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf280
        del clone_11
        del primals_39
        buf282 = buf281[0]
        buf283 = buf281[1]
        del buf281
        buf284 = buf282; del buf282  # reuse
        # Source Nodes: [x_46], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf284, convolution_17, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_17
        # Source Nodes: [x_46], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf285 = aten.convolution_backward(buf284, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_55
        del buf284
        del primals_38
        buf286 = buf285[0]
        buf287 = buf285[1]
        del buf285
        buf288 = buf273; del buf273  # reuse
        buf289 = empty((192, ), device='cuda', dtype=torch.float32)
        buf290 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_42.run(buf286, buf0, convolution_16, unsqueeze_354, squeeze_22, buf288, buf289, buf290, 192, 6272, grid=grid(192), stream=stream0)
        buf291 = buf276; del buf276  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_43.run(buf291, buf286, buf0, convolution_16, unsqueeze_354, buf289, squeeze_22, buf288, primals_36, 1204224, grid=grid(1204224), stream=stream0)
        del buf286
        del convolution_16
        del primals_36
        del squeeze_22
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf292 = aten.convolution_backward(buf291, mul_78, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_78
        del primals_35
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        buf295 = buf293; del buf293  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf295, convolution_15, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_15
        # Source Nodes: [x_41], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf296 = aten.convolution_backward(buf295, clone_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf295
        del clone_9
        del primals_34
        buf297 = buf296[0]
        buf298 = buf296[1]
        del buf296
        buf299 = buf297; del buf297  # reuse
        # Source Nodes: [x_38], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf299, convolution_14, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_14
        # Source Nodes: [x_38], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf300 = aten.convolution_backward(buf299, add_47, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_47
        del buf299
        del primals_33
        buf301 = buf300[0]
        buf302 = buf300[1]
        del buf300
        buf303 = buf289; del buf289  # reuse
        buf304 = empty((192, ), device='cuda', dtype=torch.float32)
        buf305 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf301, buf0, unsqueeze_366, squeeze_19, buf303, buf304, buf305, 192, 6272, grid=grid(192), stream=stream0)
        buf306 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_45.run(buf306, buf291, buf301, unsqueeze_366, buf304, squeeze_19, buf303, primals_31, 1204224, grid=grid(1204224), stream=stream0)
        del buf291
        del primals_31
        del squeeze_19
        del unsqueeze_366
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf307 = aten.convolution_backward(buf306, mul_65, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_65
        del primals_30
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = buf308; del buf308  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf310, convolution_12, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_12
        # Source Nodes: [x_33], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf311 = aten.convolution_backward(buf310, clone_7, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf310
        del clone_7
        del primals_29
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf314 = buf312; del buf312  # reuse
        # Source Nodes: [x_30], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf314, convolution_11, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_11
        # Source Nodes: [x_30], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf315 = aten.convolution_backward(buf314, add_39, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_39
        del buf314
        del primals_28
        buf316 = buf315[0]
        buf317 = buf315[1]
        del buf315
        buf319 = buf301; del buf301  # reuse
        # Source Nodes: [x_12, x_20, x_28], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_46.run(clone, convolution_4, convolution_7, convolution_10, unsqueeze_378, buf319, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_10
        del unsqueeze_378
        buf318 = buf304; del buf304  # reuse
        buf320 = empty((192, ), device='cuda', dtype=torch.float32)
        buf321 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf316, buf319, squeeze_16, buf318, buf320, buf321, 192, 6272, grid=grid(192), stream=stream0)
        buf322 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_48.run(buf322, buf316, buf319, buf320, squeeze_16, buf318, primals_26, 1204224, grid=grid(1204224), stream=stream0)
        del buf316
        del buf319
        del primals_26
        del squeeze_16
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf323 = aten.convolution_backward(buf322, mul_52, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_52
        del primals_25
        buf324 = buf323[0]
        buf325 = buf323[1]
        del buf323
        buf326 = buf324; del buf324  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf326, convolution_9, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_9
        # Source Nodes: [x_25], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf327 = aten.convolution_backward(buf326, clone_5, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf326
        del clone_5
        del primals_24
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = buf328; del buf328  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf330, convolution_8, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_8
        # Source Nodes: [x_22], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf331 = aten.convolution_backward(buf330, add_31, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_31
        del buf330
        del primals_23
        buf332 = buf331[0]
        buf333 = buf331[1]
        del buf331
        buf334 = buf320; del buf320  # reuse
        buf335 = empty((192, ), device='cuda', dtype=torch.float32)
        buf337 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12, x_20], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_40.run(buf332, clone, convolution_4, convolution_7, unsqueeze_390, squeeze_13, buf334, buf335, buf337, 192, 6272, grid=grid(192), stream=stream0)
        buf336 = buf332; del buf332  # reuse
        buf338 = buf322; del buf322  # reuse
        # Source Nodes: [x_12, x_20], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_41.run(buf336, buf338, clone, convolution_4, convolution_7, unsqueeze_390, buf335, squeeze_13, buf334, primals_21, 1204224, grid=grid(1204224), stream=stream0)
        del buf336
        del convolution_7
        del primals_21
        del squeeze_13
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf339 = aten.convolution_backward(buf338, mul_39, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_39
        del primals_20
        buf340 = buf339[0]
        buf341 = buf339[1]
        del buf339
        buf342 = buf340; del buf340  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf342, convolution_6, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_6
        # Source Nodes: [x_17], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf343 = aten.convolution_backward(buf342, clone_3, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf342
        del clone_3
        del primals_19
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        buf346 = buf344; del buf344  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf346, convolution_5, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_5
        # Source Nodes: [x_14], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf347 = aten.convolution_backward(buf346, add_23, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_23
        del buf346
        del primals_18
        buf348 = buf347[0]
        buf349 = buf347[1]
        del buf347
        buf350 = buf335; del buf335  # reuse
        buf351 = empty((192, ), device='cuda', dtype=torch.float32)
        buf352 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_42.run(buf348, clone, convolution_4, unsqueeze_402, squeeze_10, buf350, buf351, buf352, 192, 6272, grid=grid(192), stream=stream0)
        buf353 = buf338; del buf338  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_43.run(buf353, buf348, clone, convolution_4, unsqueeze_402, buf351, squeeze_10, buf350, primals_16, 1204224, grid=grid(1204224), stream=stream0)
        del buf348
        del convolution_4
        del primals_16
        del squeeze_10
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf354 = aten.convolution_backward(buf353, mul_26, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_26
        del primals_15
        buf355 = buf354[0]
        buf356 = buf354[1]
        del buf354
        buf357 = buf355; del buf355  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf357, convolution_3, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_3
        # Source Nodes: [x_9], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf358 = aten.convolution_backward(buf357, clone_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
        del buf357
        del clone_1
        del primals_14
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        buf361 = buf359; del buf359  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        triton_poi_fused_convolution_backward_gelu_gelu_backward_24.run(buf361, convolution_2, 2408448, grid=grid(2408448), stream=stream0)
        del convolution_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
        buf362 = aten.convolution_backward(buf361, add_15, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_15
        del buf361
        del primals_13
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf365 = buf351; del buf351  # reuse
        buf366 = empty((192, ), device='cuda', dtype=torch.float32)
        buf367 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf363, clone, unsqueeze_414, squeeze_7, buf365, buf366, buf367, 192, 6272, grid=grid(192), stream=stream0)
        buf368 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_49.run(buf368, buf363, clone, unsqueeze_414, buf366, squeeze_7, buf365, primals_11, 1204224, grid=grid(1204224), stream=stream0)
        del buf363
        del clone
        del primals_11
        del squeeze_7
        del unsqueeze_414
        buf369 = empty((1, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_50.run(buf368, buf369, 150528, 8, grid=grid(150528), stream=stream0)
        buf370 = buf366; del buf366  # reuse
        buf371 = empty((192, ), device='cuda', dtype=torch.float32)
        buf373 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf368, convolution_1, unsqueeze_426, squeeze_4, buf370, buf371, buf373, 192, 6272, grid=grid(192), stream=stream0)
        buf372 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_51.run(buf372, convolution_1, unsqueeze_426, buf371, squeeze_4, buf370, primals_9, 1204224, grid=grid(1204224), stream=stream0)
        del convolution_1
        del primals_9
        del squeeze_4
        del unsqueeze_426
        buf374 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_52.run(buf372, buf374, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf375 = aten.convolution_backward(buf372, relu, primals_7, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf372
        del primals_7
        buf376 = buf375[0]
        buf377 = buf375[1]
        del buf375
        buf378 = empty((32, 13), device='cuda', dtype=torch.float32)
        buf380 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(relu, buf376, convolution, unsqueeze_438, buf378, buf380, 416, 7720, grid=grid(416), stream=stream0)
        buf379 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_54.run(buf378, buf379, 32, 13, grid=grid(32), stream=stream0)
        del buf378
        buf381 = empty((32, ), device='cuda', dtype=torch.float32)
        buf382 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_55.run(buf380, squeeze_1, buf381, buf382, 32, 13, grid=grid(32), stream=stream0)
        del buf380
        buf383 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf383, relu, convolution, unsqueeze_438, buf381, squeeze_1, buf379, primals_5, 3211264, grid=grid(3211264), stream=stream0)
        del buf381
        del convolution
        del primals_5
        del relu
        del squeeze_1
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf384 = aten.convolution_backward(buf383, primals_206, primals_4, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf383
        del primals_206
        del primals_4
        buf385 = buf384[1]
        return (buf369, buf252, buf127, buf385, buf382, buf379, buf377, buf374, buf373, buf370, buf367, buf365, buf364, buf360, buf356, buf352, buf350, buf349, buf345, buf341, buf337, buf334, buf333, buf329, buf325, buf321, buf318, buf317, buf313, buf309, buf305, buf303, buf302, buf298, buf294, buf290, buf288, buf287, buf283, buf279, buf275, buf272, buf271, buf267, buf263, buf260, buf257, buf256, buf253, buf250, buf248, buf247, buf237, buf233, buf231, buf230, buf226, buf222, buf219, buf218, buf208, buf204, buf201, buf200, buf196, buf192, buf190, buf189, buf179, buf175, buf173, buf172, buf168, buf164, buf161, buf160, buf150, buf146, buf143, buf142, buf138, buf135, buf132, buf131, buf128, buf125, buf123, buf122, buf112, buf108, buf106, buf105, buf101, buf97, buf94, buf93, buf83, buf79, buf76, buf75, buf71, buf67, buf65, buf64, buf54, buf50, buf48, buf47, buf43, buf39, buf36, buf35, buf25, buf21, buf18, buf17, buf13, buf10, buf6, reinterpret_tensor(buf4, (1000, 768), (768, 1), 0), reinterpret_tensor(buf5, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((32, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, 32, 4, 4), (512, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 384, 2, 2), (1536, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_15 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_26 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_23 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_39 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_5 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_39 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_7 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_47 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_9 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_78 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_11 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_63 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    clone_13 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 384, 28, 28), (301056, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    add_66 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_77 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_83 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    clone_22 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_90 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_96 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    clone_30 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_103 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    clone_38 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_116 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_122 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    clone_46 = rand_strided((8, 1536, 14, 14), (301056, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    add_124 = rand_strided((8, 384, 14, 14), (75264, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_48 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_135 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_141 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone_55 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_148 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_154 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone_63 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_161 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_167 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone_71 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_174 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_180 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    clone_79 = rand_strided((8, 3072, 7, 7), (150528, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 768, 7, 7), (37632, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_81 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_30 = rand_strided((48, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    alias_9 = rand_strided((8, 6, 49, 49), (14406, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_32 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((48, 49, 128), (6272, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((48, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_38 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((8, 6, 49, 49), (14406, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_39 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((48, 49, 128), (6272, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_44 = rand_strided((48, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    alias_11 = rand_strided((8, 6, 49, 49), (14406, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_46 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((48, 49, 128), (6272, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_51 = rand_strided((48, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_52 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((8, 6, 49, 49), (14406, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((48, 128, 49), (6272, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_54 = rand_strided((48, 49, 128), (6272, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_58 = rand_strided((48, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 6, 196, 196), (230496, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((48, 196, 64), (12544, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((48, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 6, 196, 196), (230496, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((48, 196, 64), (12544, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((48, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 6, 196, 196), (230496, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((48, 196, 64), (12544, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((48, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 6, 196, 196), (230496, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_81 = rand_strided((48, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((48, 196, 64), (12544, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, convolution, squeeze_1, relu, convolution_1, squeeze_4, clone, squeeze_7, add_15, convolution_2, clone_1, convolution_3, mul_26, convolution_4, squeeze_10, add_23, convolution_5, clone_3, convolution_6, mul_39, convolution_7, squeeze_13, add_31, convolution_8, clone_5, convolution_9, mul_52, convolution_10, squeeze_16, add_39, convolution_11, clone_7, convolution_12, mul_65, convolution_13, squeeze_19, add_47, convolution_14, clone_9, convolution_15, mul_78, convolution_16, squeeze_22, add_55, convolution_17, clone_11, convolution_18, mul_91, convolution_19, squeeze_25, add_63, convolution_20, clone_13, convolution_21, mul_104, add_66, convolution_23, squeeze_28, clone_15, squeeze_31, add_77, view_7, convolution_25, squeeze_34, add_83, convolution_26, clone_22, convolution_27, squeeze_37, add_90, view_15, convolution_29, squeeze_40, add_96, convolution_30, clone_30, convolution_31, squeeze_43, add_103, view_23, convolution_33, squeeze_46, add_109, convolution_34, clone_38, convolution_35, squeeze_49, add_116, view_31, convolution_37, squeeze_52, add_122, convolution_38, clone_46, add_124, convolution_40, squeeze_55, clone_48, squeeze_58, add_135, view_39, convolution_42, squeeze_61, add_141, convolution_43, clone_55, convolution_44, squeeze_64, add_148, view_47, convolution_46, squeeze_67, add_154, convolution_47, clone_63, convolution_48, squeeze_70, add_161, view_55, convolution_50, squeeze_73, add_167, convolution_51, clone_71, convolution_52, squeeze_76, add_174, view_63, convolution_54, squeeze_79, add_180, convolution_55, clone_79, convolution_56, squeeze_82, clone_81, permute_25, unsqueeze_114, unsqueeze_126, permute_30, permute_31, alias_9, permute_32, permute_33, unsqueeze_138, unsqueeze_150, permute_37, permute_38, alias_10, permute_39, permute_40, unsqueeze_162, unsqueeze_174, permute_44, permute_45, alias_11, permute_46, permute_47, unsqueeze_186, unsqueeze_198, permute_51, permute_52, alias_12, permute_53, permute_54, unsqueeze_210, unsqueeze_222, unsqueeze_234, permute_58, permute_59, alias_13, permute_60, permute_61, unsqueeze_246, unsqueeze_258, permute_65, permute_66, alias_14, permute_67, permute_68, unsqueeze_270, unsqueeze_282, permute_72, permute_73, alias_15, permute_74, permute_75, unsqueeze_294, unsqueeze_306, permute_79, permute_80, alias_16, permute_81, permute_82, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
