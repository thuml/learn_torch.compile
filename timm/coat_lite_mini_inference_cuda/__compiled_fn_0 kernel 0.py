
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/sa/csagylybf5vxt53fsxptjwqluc77e7on55w7ea3mqr4letbkrk3a.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (48*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewu3bgiy6cuhg7bdxwjc65zguie3mwk2t5d25tu56pf3kclw2nr.py
# Source Nodes: [x1], Original ATen: [aten.native_layer_norm]
# x1 => clone, var_mean
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz26e755rd532vrge3t3hrs6yin32ajgnwzk2rhkmyfd4ancrf6z.py
# Source Nodes: [cat_39], Original ATen: [aten.cat]
# cat_39 => cat
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3137
    x1 = (xindex // 3137) % 64
    x3 = (xindex // 3137)
    x2 = (xindex // 200768)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*x3) + (((-1) + x0) % 3136)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((-1) + x0 + (3136*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr4 + ((-1) + x0 + (3136*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 64.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr5 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr6 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6symijrrxagvoyvmnoenqjt7yst5bc6xcets3kwctsd7rpsruk.py
# Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
# l__mod___serial_blocks1_0_cpe_proj => convolution_1
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (1 + x2 + (3137*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coenempq6e5jnoyvgvcm4u62cydbtaahuzc5ozjtdaroyup5wnkg.py
# Source Nodes: [cat_38, cur], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_38 => cat_1
# cur => add_3, add_4, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
triton_per_fused_cat_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp42 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((3137*r2) + (200768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*r2) + (200704*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (3137*r2) + (200768*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tmp18 - tmp28
    tmp36 = 64.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp45, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2p/c2pkyyvn33kod64bmfkiuzlzfqfhex6ur722hquxml6yup3tgpoy.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => amax, clone_1
triton_red_fused__softmax_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnx6wdryu4fvf7q243qhvzn56xsejq25xtblf2byfaqomqj22ljh.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => amax, clone_1
triton_per_fused__softmax_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinuecqewcwjp6wfnelbcz5iq3ellfywheul2jjybzsfzkceigka.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => clone_1, exp, sub_2, sum_1
triton_red_fused__softmax_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64) % 25
    x0 = xindex % 64
    x2 = (xindex // 1600)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (126*x1)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x0 + (192*r3) + (24192*x1) + (602304*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (64*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgj4lg66eyv7vbksabofwnk44qmme5wp5twz53s2hgb7jiolhysg.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => clone_1, exp, sub_2, sum_1
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (1600*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cihuizy2xokk2vxxpjexzdvngswfho6yv6qtiymh6j4f6d6lxpkk.py
# Source Nodes: [k_softmax], Original ATen: [aten._softmax]
# k_softmax => clone_1, div, exp, sub_2
triton_poi_fused__softmax_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = (xindex // 25096)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (8*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (8*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3qkjtzncwafvpaf4kp4jlug6azsrz2kpjvgpec2wdlax6yxte2.py
# Source Nodes: [factor_att], Original ATen: [aten.clone]
# factor_att => clone_2
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
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrkrlnvup2xvpbabutgz5mrnxdshpgdjrzikkp6yrydur52svbx.py
# Source Nodes: [factor_att_1], Original ATen: [aten.clone]
# factor_att_1 => clone_3
triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (192*x1) + (602304*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgjabzdkcw6i3gzw6dujqjgrqpl3j7kqcztzxpbghlq5eezbr2a.py
# Source Nodes: [x_10], Original ATen: [aten.clone]
# x_10 => clone_4
triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = (xindex // 8) % 3137
    x0 = xindex % 8
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (8*x2) + (192*x1) + (602304*x3)), tmp5 & xmask, other=0.0)
    tmp7 = x0 + (8*x2)
    tmp8 = tmp7 >= tmp4
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp7 < tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr2 + ((3136*x0) + (25088*x2) + (50176*x3) + (((-1) + x1) % 3136)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (8*x2)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp7 >= tmp9
    tmp18 = tl.full([1], 40, tl.int64)
    tmp19 = tmp7 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr4 + ((-50176) + (3136*x0) + (25088*x2) + (75264*x3) + (((-1) + x1) % 3136)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-16) + x0 + (8*x2)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp7 >= tmp18
    tmp28 = tl.full([1], 64, tl.int64)
    tmp29 = tmp7 < tmp28
    tmp30 = tmp27 & tmp5
    tmp31 = tl.load(in_ptr6 + ((-125440) + (3136*x0) + (25088*x2) + (75264*x3) + (((-1) + x1) % 3136)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-40) + x0 + (8*x2)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp20, tmp26, tmp35)
    tmp37 = tl.where(tmp10, tmp16, tmp36)
    tmp38 = tmp6 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tmp2 + tmp40
    tl.store(out_ptr0 + (x0 + (8*x2) + (64*x1) + (200768*x3)), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/ciovcc4lewtc3pb5jnbjsvpjzmv47vx6lghg2paoffqqclr5o7oe.py
# Source Nodes: [cat_38, cur_2, x_13], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_38 => cat_1
# cur_2 => add_7, add_8, mul_6, mul_7, rsqrt_2, sub_3, var_mean_2
# x_13 => add_6
triton_per_fused_add_cat_native_layer_norm_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3137
    r2 = rindex
    x1 = (xindex // 3137)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((3137*r2) + (200768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*r2) + (200704*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (3137*r2) + (200768*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp22 - tmp32
    tmp40 = 64.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(out_ptr0 + (x0 + (3137*r2) + (200768*x1)), tmp22, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ces2tzyndncirmbjjimvaesaitu5fv3yr3uz7pb3m5marxadjvgv.py
# Source Nodes: [x_16], Original ATen: [aten.gelu]
# x_16 => add_9, erf, mul_10, mul_8, mul_9
triton_poi_fused_gelu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12849152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmqq2janhv35txpx4jlwe7rrkj46uti4yb6mw3l6h5zmzehml4u.py
# Source Nodes: [x1_2], Original ATen: [aten.add]
# x1_2 => add_10
triton_poi_fused_add_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3137
    y1 = (yindex // 3137)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3137*x2) + (200768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/csetsuhvgrbnmciyz36dus4cl76vuva3vajrvkqypw6evjlezwk7.py
# Source Nodes: [cat_36, cur_4], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_36 => cat_3
# cur_4 => var_mean_3
triton_per_fused_cat_native_layer_norm_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3137
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 3137)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*r2) + (200704*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhhpizg7r27rzcsw4e2v23pl7otu6ttya3y2sgmnwcs2favft5q.py
# Source Nodes: [cat_36, cur_4], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_36 => cat_3
# cur_4 => add_12, add_13, mul_11, mul_12, rsqrt_3, sub_4, var_mean_3
triton_poi_fused_cat_native_layer_norm_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3137
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
    tmp19 = tl.load(in_ptr3 + (x2 + (3137*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2 + (3137*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (64*x2) + (200768*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*y3) + (((-1) + x2) % 3136)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (64 + y0 + (64*(((-1) + x2) % 3136)) + (200768*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 64.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (64*x2) + (200768*y1)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3d5xysbqfn4seea4qqqv5lihovslv7oaq5t6kxdlvijqh3zjwsp.py
# Source Nodes: [cat_36, cur_6, x_31], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_36 => cat_3
# cur_6 => add_16, add_17, mul_15, mul_16, rsqrt_4, sub_6, var_mean_4
# x_31 => add_15
triton_per_fused_add_cat_native_layer_norm_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3137
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 3137)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 3137, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((3136*r2) + (200704*x1) + (((-1) + x0) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (64 + r2 + (64*(((-1) + x0) % 3136)) + (200768*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp22 - tmp32
    tmp40 = 64.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosxyintg4zcbla4uhmw3n4ttzs25z5iu3jnqml2w32rhnjrutoc.py
# Source Nodes: [x1_nocls], Original ATen: [aten.clone]
# x1_nocls => clone_15
triton_poi_fused_clone_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 200704)
    x3 = xindex % 200704
    x0 = xindex % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x3 + (200768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (64 + x3 + (200768*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrsuvuedlifndwnslsoau5zzyqs25pmaxynifgerlkkautmvhgt.py
# Source Nodes: [x1_nocls, x_40], Original ATen: [aten.clone, aten.convolution]
# x1_nocls => clone_15
# x_40 => convolution_9
triton_poi_fused_clone_convolution_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (256*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zj/czjvqdsl5yk4kqgewhtxri24xqnqrd35yqnnppwdszotknafvasv.py
# Source Nodes: [x2], Original ATen: [aten.native_layer_norm]
# x2 => clone_16, var_mean_5
triton_red_fused_native_layer_norm_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tp/ctpaqhtkuuc5r42uicmobnr4hqngcwmthm6rhyen6zhpqzpsyuti.py
# Source Nodes: [cat_34], Original ATen: [aten.cat]
# cat_34 => cat_5
triton_poi_fused_cat_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 785
    x1 = (xindex // 785) % 128
    x3 = (xindex // 785)
    x2 = (xindex // 100480)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*x3) + (((-1) + x0) % 784)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((-1) + x0 + (784*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr4 + ((-1) + x0 + (784*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 128.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr5 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr6 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnwq6mbkkv4aiyqv4sbqobs6op22hws6kewujavaos4nhayc5tk.py
# Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
# l__mod___serial_blocks2_0_cpe_proj => convolution_10
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (1 + x2 + (785*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyuhzpssoiw6tsjbhgipsibcomsi2twenhat66llbjwnv4h56iy.py
# Source Nodes: [cat_33, cur_8], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_33 => cat_6
# cur_8 => add_23, add_24, mul_22, mul_23, rsqrt_6, sub_8, var_mean_6
triton_red_fused_cat_native_layer_norm_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((785*r2) + (100480*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 785, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr0 + (1 + (785*r2) + (100480*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp8, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp7, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp49 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = x0
        tmp24 = tl.full([1, 1], 0, tl.int64)
        tmp25 = tmp23 >= tmp24
        tmp26 = tl.full([1, 1], 1, tl.int64)
        tmp27 = tmp23 < tmp26
        tmp28 = tl.load(in_ptr0 + ((785*r2) + (100480*x1)), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
        tmp30 = tl.where(tmp27, tmp28, tmp29)
        tmp31 = tmp23 >= tmp26
        tmp32 = tl.full([1, 1], 785, tl.int64)
        tmp33 = tmp23 < tmp32
        tmp34 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 + tmp35
        tmp37 = tl.load(in_ptr0 + (1 + (785*r2) + (100480*x1) + (((-1) + x0) % 784)), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tmp36 + tmp37
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp31, tmp38, tmp39)
        tmp41 = tl.where(tmp27, tmp30, tmp40)
        tmp42 = tmp41 - tmp20
        tmp43 = 128.0
        tmp44 = tmp21 / tmp43
        tmp45 = 1e-06
        tmp46 = tmp44 + tmp45
        tmp47 = tl.math.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp48 * tmp49
        tmp52 = tmp50 + tmp51
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp52, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csjj6ixwvzknyoxwwdwdvepo6hlb3aga7usmldjodnb4g2ewyl2p.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => amax_2, clone_17
triton_red_fused__softmax_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbpuvbn7ibslhgy4pnjlg5gq7g35zvb24hfw3n3mtcf4eemeody.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => amax_2, clone_17
triton_per_fused__softmax_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cmeqymw777c35sjzmhvqrq2uvxxrbut4ffxddf65gbcffwt6bog5.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => clone_17, exp_2, sub_9, sum_3
triton_red_fused__softmax_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128) % 7
    x0 = xindex % 128
    x2 = (xindex // 896)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (113*x1)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x0 + (384*r3) + (43392*x1) + (301440*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (128*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ui/cuiftygsipegywqvuu4l3g3lh55nncwroxrbmup6nwjt3xpqbyir.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => clone_17, exp_2, sub_9, sum_3
triton_per_fused__softmax_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (896*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinkqg275fjjnkeoicjmoubid6kbynpuq5w5yb3dtssdnle6kvot.py
# Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
# k_softmax_2 => clone_17, div_2, exp_2, sub_9
triton_poi_fused__softmax_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = (xindex // 12560)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (16*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4lpjm3q6gitdiawgu6jtu7yvlfq7654acq5gbusel2ebzwbkc4.py
# Source Nodes: [factor_att_4], Original ATen: [aten.clone]
# factor_att_4 => clone_18
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmn2h6cj6yqkd7uzg5telmhx6nzhphg76msr4sxmud47rav4x32b.py
# Source Nodes: [factor_att_5], Original ATen: [aten.clone]
# factor_att_5 => clone_19
triton_poi_fused_clone_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (384*x1) + (301440*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaatedwiouilauv6rvwhthyic6w5x6w75mu2xorzsmoz26ergif.py
# Source Nodes: [x_50], Original ATen: [aten.clone]
# x_50 => clone_20
triton_poi_fused_clone_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = (xindex // 16) % 785
    x0 = xindex % 16
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (16*x2) + (384*x1) + (301440*x3)), tmp5 & xmask, other=0.0)
    tmp7 = x0 + (16*x2)
    tmp8 = tmp7 >= tmp4
    tmp9 = tl.full([1], 32, tl.int64)
    tmp10 = tmp7 < tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr2 + ((784*x0) + (12544*x2) + (25088*x3) + (((-1) + x1) % 784)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (16*x2)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp7 >= tmp9
    tmp18 = tl.full([1], 80, tl.int64)
    tmp19 = tmp7 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr4 + ((-25088) + (784*x0) + (12544*x2) + (37632*x3) + (((-1) + x1) % 784)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-32) + x0 + (16*x2)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp7 >= tmp18
    tmp28 = tl.full([1], 128, tl.int64)
    tmp29 = tmp7 < tmp28
    tmp30 = tmp27 & tmp5
    tmp31 = tl.load(in_ptr6 + ((-62720) + (784*x0) + (12544*x2) + (37632*x3) + (((-1) + x1) % 784)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-80) + x0 + (16*x2)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp20, tmp26, tmp35)
    tmp37 = tl.where(tmp10, tmp16, tmp36)
    tmp38 = tmp6 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tmp2 + tmp40
    tl.store(out_ptr0 + (x0 + (16*x2) + (128*x1) + (100480*x3)), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tavwcobnacexilbn2onm52rmmziigoayqs2fdvtdml2dc64u6j.py
# Source Nodes: [cat_33, cur_10, x_53], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_33 => cat_6
# cur_10 => add_27, add_28, mul_26, mul_27, rsqrt_7, sub_10, var_mean_7
# x_53 => add_26
triton_red_fused_add_cat_native_layer_norm_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 785
    x1 = (xindex // 785)
    x3 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((785*r2) + (100480*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 785, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr0 + (1 + (785*r2) + (100480*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp8, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp7, tmp17)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp18 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
        tl.store(out_ptr0 + (x0 + (785*r2) + (100480*x1)), tmp22, rmask & xmask)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp29_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(out_ptr0 + (x0 + (785*r2) + (100480*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp29_mean_next, tmp29_m2_next, tmp29_weight_next = triton_helpers.welford_reduce(
            tmp28, tmp29_mean, tmp29_m2, tmp29_weight,
        )
        tmp29_mean = tl.where(rmask & xmask, tmp29_mean_next, tmp29_mean)
        tmp29_m2 = tl.where(rmask & xmask, tmp29_m2_next, tmp29_m2)
        tmp29_weight = tl.where(rmask & xmask, tmp29_weight_next, tmp29_weight)
    tmp29_tmp, tmp30_tmp, tmp31_tmp = triton_helpers.welford(
        tmp29_mean, tmp29_m2, tmp29_weight, 1
    )
    tmp29 = tmp29_tmp[:, None]
    tmp30 = tmp30_tmp[:, None]
    tmp31 = tmp31_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(out_ptr0 + (x0 + (785*r2) + (100480*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32 - tmp24
        tmp34 = 128.0
        tmp35 = tmp30 / tmp34
        tmp36 = 1e-06
        tmp37 = tmp35 + tmp36
        tmp38 = tl.math.rsqrt(tmp37)
        tmp39 = tmp33 * tmp38
        tmp41 = tmp39 * tmp40
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbupsxt7y5w3e4yqg4t7xaqatd5ox46rmdbuscwxpi22gzml6wst.py
# Source Nodes: [x_56], Original ATen: [aten.gelu]
# x_56 => add_29, erf_2, mul_28, mul_29, mul_30
triton_poi_fused_gelu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6430720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2e5rbt77a6j25kzjqvih7uqsigsfn72nlwgbld3mkaop3y26md2.py
# Source Nodes: [x2_2], Original ATen: [aten.add]
# x2_2 => add_30
triton_poi_fused_add_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6280
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 785
    y1 = (yindex // 785)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (785*x2) + (100480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coexd6yt75bh4lmvrnnp5ro6lcq3estetr2fxpjtkjxk47evvuda.py
# Source Nodes: [cat_31, cur_12], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_31 => cat_8
# cur_12 => var_mean_8
triton_per_fused_cat_native_layer_norm_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 785)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktizj35xfor6ctlr6np523olrnsmyzdxgur7nilzr5tg35fmxdx.py
# Source Nodes: [cat_31, cur_12], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_31 => cat_8
# cur_12 => add_32, add_33, mul_31, mul_32, rsqrt_8, sub_11, var_mean_8
triton_poi_fused_cat_native_layer_norm_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 785
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
    tmp19 = tl.load(in_ptr3 + (x2 + (785*y1)), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2 + (785*y1)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (128*x2) + (100480*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*y3) + (((-1) + x2) % 784)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (128 + y0 + (128*(((-1) + x2) % 784)) + (100480*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 128.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (128*x2) + (100480*y1)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cnty2dtqm647hpruuctrclzguzqkoptjdy2ps4skdym5ivdjjrar.py
# Source Nodes: [cat_31, cur_14, x_71], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_31 => cat_8
# cur_14 => add_36, add_37, mul_35, mul_36, rsqrt_9, sub_13, var_mean_9
# x_71 => add_35
triton_per_fused_add_cat_native_layer_norm_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 785)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (128 + r2 + (128*(((-1) + x0) % 784)) + (100480*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp30 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp22 - tmp32
    tmp40 = 128.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4z3aj4iqcy774pvzkaup4kov45veeghgdmy35b7xmel77p7zk3r.py
# Source Nodes: [x2_nocls], Original ATen: [aten.clone]
# x2_nocls => clone_31
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 100352)
    x3 = xindex % 100352
    x0 = xindex % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x3 + (100480*x2)), None)
    tmp1 = tl.load(in_ptr1 + (128 + x3 + (100480*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxboigwrinxk4pnsrv4u64uq3uzilj5kv7nv4u4bu7k722cuzpn.py
# Source Nodes: [x2_nocls, x_80], Original ATen: [aten.clone, aten.convolution]
# x2_nocls => clone_31
# x_80 => convolution_18
triton_poi_fused_clone_convolution_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (512*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lsemw2gqlcdicifdqmwfr23ee5xvznyrcaptrigbnrhpsvzflg.py
# Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
# x3 => clone_32, var_mean_10
triton_red_fused_native_layer_norm_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 196) % 3
    x0 = xindex % 196
    x2 = (xindex // 588)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (196*r3) + (20972*x1) + (62720*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r3 + (107*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = 1.0
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_combine(
            tmp17_mean, tmp17_m2, tmp17_weight,
            tmp14, tmp15, tmp16
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp17, xmask)
    tl.store(out_ptr1 + (x4), tmp18, xmask)
    tl.store(out_ptr2 + (x4), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmse3jpv6eumr7lbyinel6cjgis65b6hw5fsg2xonkmlt3ty6mcc.py
# Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
# x3 => clone_32, var_mean_10
triton_per_fused_native_layer_norm_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafumwa3aoivw446galrd5fm4j6wk3hsm5sr53754yzyxv7qy4rh.py
# Source Nodes: [cat_29], Original ATen: [aten.cat]
# cat_29 => cat_10
triton_poi_fused_cat_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 197
    x1 = (xindex // 197) % 320
    x3 = (xindex // 197)
    x2 = (xindex // 63040)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x3) + (((-1) + x0) % 196)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((-1) + x0 + (196*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr4 + ((-1) + x0 + (196*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 320.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr5 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr6 + (x1), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdylxqbk3misd2wkixgh5eahh2j6lmoojcngygne4pcp73xhnmcb.py
# Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
# l__mod___serial_blocks3_0_cpe_proj => convolution_19
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
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
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (1 + x2 + (197*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (62720*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtbixsg76orsbiui3pnby4rmruptdsswlb2hsizu2zontz537xt.py
# Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_28 => cat_11
# cur_16 => var_mean_11
triton_red_fused_cat_native_layer_norm_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4728
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 197) % 3
    x0 = xindex % 197
    x2 = (xindex // 591)
    tmp35_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp35_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp35_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 1, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((197*r3) + (21079*x1) + (63040*x2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 197, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp2
        tmp16 = tl.load(in_ptr1 + ((196*r3) + (20972*x1) + (62720*x2) + (((-1) + x0) % 196)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r3 + (107*x1)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.load(in_ptr0 + (1 + (197*r3) + (21079*x1) + (63040*x2) + (((-1) + x0) % 196)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp15, tmp20, tmp21)
        tmp23 = tl.where(tmp7, tmp11, tmp22)
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = 0.0
        tmp27 = tl.full(tmp26.shape, 0, tmp26.dtype)
        tmp28 = tl.where(tmp2, tmp26, tmp27)
        tmp29 = 1.0
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp33 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp34 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp35_mean_next, tmp35_m2_next, tmp35_weight_next = triton_helpers.welford_combine(
            tmp35_mean, tmp35_m2, tmp35_weight,
            tmp32, tmp33, tmp34
        )
        tmp35_mean = tl.where(rmask & xmask, tmp35_mean_next, tmp35_mean)
        tmp35_m2 = tl.where(rmask & xmask, tmp35_m2_next, tmp35_m2)
        tmp35_weight = tl.where(rmask & xmask, tmp35_weight_next, tmp35_weight)
    tmp35_tmp, tmp36_tmp, tmp37_tmp = triton_helpers.welford(
        tmp35_mean, tmp35_m2, tmp35_weight, 1
    )
    tmp35 = tmp35_tmp[:, None]
    tmp36 = tmp36_tmp[:, None]
    tmp37 = tmp37_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp35, xmask)
    tl.store(out_ptr1 + (x4), tmp36, xmask)
    tl.store(out_ptr2 + (x4), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclpujelwbdbdg6i6xv3yxaxt2u6g2fgqy35cvvud7cxd36cijzd.py
# Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_28 => cat_11
# cur_16 => var_mean_11
triton_per_fused_cat_native_layer_norm_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (197*r2) + (591*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypyjrq3psgekbh52z5tlif5ndcm5kn6wgiswjuznxu62exntwqw.py
# Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_28 => cat_11
# cur_16 => add_43, add_44, mul_42, mul_43, rsqrt_11, sub_15, var_mean_11
triton_poi_fused_cat_native_layer_norm_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 320) % 197
    x0 = xindex % 320
    x2 = (xindex // 63040)
    x3 = (xindex // 320)
    x4 = xindex
    tmp19 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((197*x0) + (63040*x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x0) + (62720*x2) + (((-1) + x1) % 196)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (197*x0) + (63040*x2) + (((-1) + x1) % 196)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 320.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cuepjhhuftqihtvagll454ksmujfetcp3xzedcly5dz2sw5mcmxp.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => amax_4, clone_33
triton_red_fused__softmax_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, float("-inf"), tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccmtm6ycvfl5p5wuytbibtuyt5hov2tm6omcgrnbbqmngcbamlc.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => amax_4, clone_33
triton_per_fused__softmax_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2lpitqtspzmps27pvyfnmxbuopwkigvjcru73wua7ncyut72cc.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => clone_33, exp_4, sub_16, sum_5
triton_red_fused__softmax_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320) % 2
    x0 = xindex % 320
    x2 = (xindex // 640)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (99*x1)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (960*r3) + (95040*x1) + (189120*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0 + (320*x2), [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2ka3bghy66r4dqfsvzsn2u7kbn3mboaswpjqm7mjrrmyckexuqg.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => clone_33, exp_4, sub_16, sum_5
triton_per_fused__softmax_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 320
    x1 = (xindex // 320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r2) + (640*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwa3ehk6v5njawz3xd3kz5utv2yvaku7ohlwefpfuwunfjtcljg.py
# Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
# k_softmax_4 => clone_33, div_4, exp_4, sub_16
triton_poi_fused__softmax_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = (xindex // 7880)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (40*x4)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (40*x4)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpv5atpeul4vqs5c4bjyyrb7xq5l56n7idolcentv6uf4bp6ky5j.py
# Source Nodes: [factor_att_8], Original ATen: [aten.clone]
# factor_att_8 => clone_34
triton_poi_fused_clone_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (640 + x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklhngywqeopagfhxcvbmaxfmjw7ghgvfwz76ovlbsoerd4bdr2j.py
# Source Nodes: [factor_att_9], Original ATen: [aten.clone]
# factor_att_9 => clone_35
triton_poi_fused_clone_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*x2) + (960*x1) + (189120*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyz55l7wtw3cb6zx3in46kfmrxuqftnkov6x5zycahtopa5raerj.py
# Source Nodes: [x_90], Original ATen: [aten.clone]
# x_90 => clone_36
triton_poi_fused_clone_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = (xindex // 40) % 197
    x0 = xindex % 40
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = 0.15811388300841897
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (40*x2) + (960*x1) + (189120*x3)), tmp5 & xmask, other=0.0)
    tmp7 = x0 + (40*x2)
    tmp8 = tmp7 >= tmp4
    tmp9 = tl.full([1], 80, tl.int64)
    tmp10 = tmp7 < tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr2 + ((196*x0) + (7840*x2) + (15680*x3) + (((-1) + x1) % 196)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (40*x2)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp7 >= tmp9
    tmp18 = tl.full([1], 200, tl.int64)
    tmp19 = tmp7 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr4 + ((-15680) + (196*x0) + (7840*x2) + (23520*x3) + (((-1) + x1) % 196)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-80) + x0 + (40*x2)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp7 >= tmp18
    tmp28 = tl.full([1], 320, tl.int64)
    tmp29 = tmp7 < tmp28
    tmp30 = tmp27 & tmp5
    tmp31 = tl.load(in_ptr6 + ((-39200) + (196*x0) + (7840*x2) + (23520*x3) + (((-1) + x1) % 196)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-200) + x0 + (40*x2)), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp20, tmp26, tmp35)
    tmp37 = tl.where(tmp10, tmp16, tmp36)
    tmp38 = tmp6 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tmp2 + tmp40
    tl.store(out_ptr0 + (x0 + (40*x2) + (320*x1) + (63040*x3)), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgirldvy3jeib6jgwtd4gtshc73olczu5dr36utrgez5snw4xtba.py
# Source Nodes: [cat_28, x_93], Original ATen: [aten.add, aten.cat]
# cat_28 => cat_11
# x_93 => add_46
triton_poi_fused_add_cat_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 197
    x2 = xindex
    y1 = (yindex // 197)
    y3 = yindex
    tmp19 = tl.load(in_ptr3 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((197*x2) + (63040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*x2) + (62720*y1) + (((-1) + y0) % 196)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (197*x2) + (63040*y1) + (((-1) + y0) % 196)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (y0 + (197*x2) + (63040*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxub3ygtgnr4ocwlcsuhiim5rejj3ju47hs2vn3u7edlgddpjyl.py
# Source Nodes: [cur_18], Original ATen: [aten.native_layer_norm]
# cur_18 => var_mean_12
triton_red_fused_native_layer_norm_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4728
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 197) % 3
    x0 = xindex % 197
    x2 = (xindex // 591)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (197*r3) + (21079*x1) + (63040*x2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp15, xmask)
    tl.store(out_ptr1 + (x4), tmp16, xmask)
    tl.store(out_ptr2 + (x4), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czphy6pouaoi5bzpgu6apb7jx7kwfd5x5c4x2uj5bnqrxg3dbryt.py
# Source Nodes: [cur_18], Original ATen: [aten.native_layer_norm]
# cur_18 => add_47, add_48, mul_46, mul_47, rsqrt_12, sub_17, var_mean_12
triton_poi_fused_native_layer_norm_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 197
    y1 = (yindex // 197)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (197*x2) + (63040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 320.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (320*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7e23ubzcwlst7wsuqoyjxrwfhddlozqewsefrye2jnp33wl3bt6.py
# Source Nodes: [x_96], Original ATen: [aten.gelu]
# x_96 => add_49, erf_4, mul_48, mul_49, mul_50
triton_poi_fused_gelu_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2017280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqobss7hhtaqc2462mi24ejhmq4dv42ki4m3wntne4oep3rbfaa2.py
# Source Nodes: [x3_2], Original ATen: [aten.add]
# x3_2 => add_50
triton_poi_fused_add_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 320
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 197
    y1 = (yindex // 197)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (197*x2) + (63040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (320*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (320*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cunslo4twd4ry4ykdw56lbcqyvmwiep4h2tnktuasx7gaqyvfjfl.py
# Source Nodes: [cat_26, cur_20], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_26 => cat_13
# cur_20 => var_mean_13
triton_per_fused_cat_native_layer_norm_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 197)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (320*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*r2) + (62720*x1) + (((-1) + x0) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 320, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4yqwfoej44b2jd3m55qsv6t55awcfq5iarvgmok3o2oxvb4w2t.py
# Source Nodes: [cat_26, cur_20], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_26 => cat_13
# cur_20 => add_52, add_53, mul_51, mul_52, rsqrt_13, sub_18, var_mean_13
triton_poi_fused_cat_native_layer_norm_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 197
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
    tmp19 = tl.load(in_ptr3 + (x2 + (197*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2 + (197*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (320*x2) + (63040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*y3) + (((-1) + x2) % 196)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (320 + y0 + (320*(((-1) + x2) % 196)) + (63040*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 320.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (320*x2) + (63040*y1)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6vr3chhh4fw433dvxt3gxe3v4jh6z4bmhvlvsorwq7rtxvnwvc.py
# Source Nodes: [cat_26, cur_22, x_111], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_26 => cat_13
# cur_22 => add_56, add_57, mul_55, mul_56, rsqrt_14, sub_20, var_mean_14
# x_111 => add_55
triton_per_fused_add_cat_native_layer_norm_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 197)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (320*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 197, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((196*r2) + (62720*x1) + (((-1) + x0) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (320 + r2 + (320*(((-1) + x0) % 196)) + (63040*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 320, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 320.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (320*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqj6nu5nrtxgahy5vpdfojtjos6djjvif7dzdpb65mya7wgyl6h.py
# Source Nodes: [x3_nocls], Original ATen: [aten.clone]
# x3_nocls => clone_47
triton_poi_fused_clone_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 62720)
    x3 = xindex % 62720
    x0 = xindex % 320
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x3 + (63040*x2)), None)
    tmp1 = tl.load(in_ptr1 + (320 + x3 + (63040*x2)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ka/ckanbtzb3syxyec5kyn5g4cw4qmfdqywomr2glcjodg7rbcshfgc.py
# Source Nodes: [x3_nocls, x_120], Original ATen: [aten.clone, aten.convolution]
# x3_nocls => clone_47
# x_120 => convolution_27
triton_poi_fused_clone_convolution_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 163840
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + (4*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (1280*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4r/c4rdtvzlxusrjulob2i6jgp4i2uloutbvaxnx3jdu5ibixduhfjv.py
# Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
# x4 => clone_48, var_mean_15
triton_red_fused_native_layer_norm_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 4
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (49*r3) + (6272*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbwzslcryj2a64dqfvfxube5pughs6sh6eyizkflp63kdbcd2en.py
# Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
# x4 => clone_48, var_mean_15
triton_per_fused_native_layer_norm_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 49
    x1 = (xindex // 49)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (49*r2) + (196*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l3/cl36crly4w6rzamuckv3mcgwvzqnastlb52jhndn2dh4cxbzhzxr.py
# Source Nodes: [cat_24], Original ATen: [aten.cat]
# cat_24 => cat_15
triton_poi_fused_cat_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50
    x1 = (xindex // 50) % 512
    x3 = (xindex // 50)
    x2 = (xindex // 25600)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*x3) + (((-1) + x0) % 49)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr3 + ((-1) + x0 + (49*x2)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr4 + ((-1) + x0 + (49*x2)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr5 + (x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr6 + (x1), tmp8, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbku6iwmcxnrqv4jjsucduizacsjmsrdh7ucktnjqoc43iv6piy.py
# Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
# l__mod___serial_blocks4_0_cpe_proj => convolution_28
triton_poi_fused_convolution_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (1 + x2 + (50*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (25088*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5bnbtdc6375npdzf7dcoimwzgebooxyyw4ovjj6w6ldqa52xqw.py
# Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_23 => cat_16
# cur_24 => var_mean_16
triton_red_fused_cat_native_layer_norm_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1600
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 50
    x4 = (xindex // 50)
    x1 = (xindex // 50) % 4
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((50*r3) + (6400*x4)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 50, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((49*r3) + (6272*x4) + (((-1) + x0) % 49)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr0 + (1 + (50*r3) + (6400*x4) + (((-1) + x0) % 49)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp8, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp7, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp20, xmask)
    tl.store(out_ptr1 + (x5), tmp21, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4ffl2adafibvfgvyisxbvz77icuykedswfvl2ff3num2chizm4s.py
# Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_23 => cat_16
# cur_24 => var_mean_16
triton_per_fused_cat_native_layer_norm_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 50
    x1 = (xindex // 50)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50*r2) + (200*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (50*r2) + (200*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (50*r2) + (200*x1)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfg2ib2nz55erxbqktivrjnsscdn4p6ooh7r5chkt3alpofhrguu.py
# Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_23 => cat_16
# cur_24 => add_63, add_64, mul_62, mul_63, rsqrt_16, sub_22, var_mean_16
triton_poi_fused_cat_native_layer_norm_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 50
    x0 = xindex % 512
    x2 = (xindex // 25600)
    x3 = (xindex // 512)
    x4 = xindex
    tmp19 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((50*x0) + (25600*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*x0) + (25088*x2) + (((-1) + x1) % 49)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (50*x0) + (25600*x2) + (((-1) + x1) % 49)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2smuslqejjgeqwlqygm2fix6ix6xwun45e2jyfrgq77vixzn3nl.py
# Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
# k_softmax_6 => amax_6, clone_49, exp_6, sub_23, sum_7
triton_per_fused__softmax_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1536*r2) + (76800*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fdzilnnn5qykqvidrb3fgt5t4j52suzkupqmrprw4yst5hdnh7.py
# Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
# k_softmax_6 => clone_49, div_6, exp_6, sub_23
triton_poi_fused__softmax_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = (xindex // 3200)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 + (64*x4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.exp(tmp2)
    tmp5 = tmp3 / tmp4
    tl.store(out_ptr0 + (x5), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jmpbzlpa6ei4vuypqcagximqvbifmhjdbhc4lksm3cdv77b2kf.py
# Source Nodes: [factor_att_12], Original ATen: [aten.clone]
# factor_att_12 => clone_50
triton_poi_fused_clone_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3aouhr45mjhirtljjwaz3egm6fchog5hsnuyth37to3s74gspi.py
# Source Nodes: [factor_att_13], Original ATen: [aten.clone]
# factor_att_13 => clone_51
triton_poi_fused_clone_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1) + (76800*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq2kgtyplonltmnxxje6wwscapfcpsbh62bk66haopjetcunsss.py
# Source Nodes: [x_130], Original ATen: [aten.clone]
# x_130 => clone_52
triton_poi_fused_clone_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = (xindex // 64) % 50
    x0 = xindex % 64
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (64*x2) + (1536*x1) + (76800*x3)), tmp5, other=0.0)
    tmp7 = x0 + (64*x2)
    tmp8 = tmp7 >= tmp4
    tmp9 = tl.full([1], 128, tl.int64)
    tmp10 = tmp7 < tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr2 + ((49*x0) + (3136*x2) + (6272*x3) + (((-1) + x1) % 49)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (x0 + (64*x2)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp7 >= tmp9
    tmp18 = tl.full([1], 320, tl.int64)
    tmp19 = tmp7 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tmp20 & tmp5
    tmp22 = tl.load(in_ptr4 + ((-6272) + (49*x0) + (3136*x2) + (9408*x3) + (((-1) + x1) % 49)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr5 + ((-128) + x0 + (64*x2)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tmp7 >= tmp18
    tmp28 = tl.full([1], 512, tl.int64)
    tmp29 = tmp7 < tmp28
    tmp30 = tmp27 & tmp5
    tmp31 = tl.load(in_ptr6 + ((-15680) + (49*x0) + (3136*x2) + (9408*x3) + (((-1) + x1) % 49)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr7 + ((-320) + x0 + (64*x2)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp20, tmp26, tmp35)
    tmp37 = tl.where(tmp10, tmp16, tmp36)
    tmp38 = tmp6 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp41 = tmp2 + tmp40
    tl.store(out_ptr0 + (x0 + (64*x2) + (512*x1) + (25600*x3)), tmp41, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlwr2b3t2de3qcxueemzdqj7n5agkmv5p6jfi7b4azzsc3ptdaz.py
# Source Nodes: [cat_23, x_133], Original ATen: [aten.add, aten.cat]
# cat_23 => cat_16
# x_133 => add_66
triton_poi_fused_add_cat_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 400
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 50
    x2 = xindex
    y1 = (yindex // 50)
    y3 = yindex
    tmp19 = tl.load(in_ptr3 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((50*x2) + (25600*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*x2) + (25088*y1) + (((-1) + y0) % 49)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(x2, [XBLOCK, YBLOCK])), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (1 + (50*x2) + (25600*y1) + (((-1) + y0) % 49)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (y0 + (50*x2) + (25600*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4t3c6cvklbrp7kmr4x34avqe63nus6nfd6l3j5p52wqvb3dus7d.py
# Source Nodes: [cur_26], Original ATen: [aten.native_layer_norm]
# cur_26 => var_mean_17
triton_red_fused_native_layer_norm_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1600
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 50
    x1 = (xindex // 50)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (50*r2) + (6400*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/cox2346i4bblmhlpdmd7xx5zdhirhy2j75zopojxypurjynfxlx5.py
# Source Nodes: [cur_26], Original ATen: [aten.native_layer_norm]
# cur_26 => add_67, add_68, mul_66, mul_67, rsqrt_17, sub_24, var_mean_17
triton_poi_fused_native_layer_norm_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 400
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 50
    y1 = (yindex // 50)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (50*x2) + (25600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hoig24qdvawzcbe4u76xmfmzus3sridc52vljenxkoo7bt73ns.py
# Source Nodes: [x_136], Original ATen: [aten.gelu]
# x_136 => add_69, erf_6, mul_68, mul_69, mul_70
triton_poi_fused_gelu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77gz7bs24pikhrjyk6xlhuz7jsryzucanei3brpjrhm2irhidtg.py
# Source Nodes: [x4_2], Original ATen: [aten.add]
# x4_2 => add_70
triton_poi_fused_add_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 400
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 50
    y1 = (yindex // 50)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (50*x2) + (25600*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvj2nypdqm3rbnsemanehpiugxxcyairxjpekoghll45ua5uh7n2.py
# Source Nodes: [cat_21, cur_28], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_21 => cat_18
# cur_28 => var_mean_18
triton_per_fused_cat_native_layer_norm_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 50
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 50)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*r2) + (25088*x1) + (((-1) + x0) % 49)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewva4zavwysatp2gsk5xhzwg4g7gxiw4m4tajxuovbnc7irtyzj.py
# Source Nodes: [cat_21, cur_28], Original ATen: [aten.cat, aten.native_layer_norm]
# cat_21 => cat_18
# cur_28 => add_72, add_73, mul_71, mul_72, rsqrt_18, sub_25, var_mean_18
triton_poi_fused_cat_native_layer_norm_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_layer_norm_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 50
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
    tmp19 = tl.load(in_ptr3 + (x2 + (50*y1)), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2 + (50*y1)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (512*x2) + (25600*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*y3) + (((-1) + x2) % 49)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (512 + y0 + (512*(((-1) + x2) % 49)) + (25600*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp20 = tmp18 - tmp19
    tmp22 = 512.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp20 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (y0 + (512*x2) + (25600*y1)), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqv7oa5k7ksdrkq76nbncyofnue4rhw75auusxgvjmw2jhu3zbm5.py
# Source Nodes: [cat_21, cur_30, x_151], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
# cat_21 => cat_18
# cur_30 => add_76, add_77, mul_75, mul_76, rsqrt_19, sub_27, var_mean_19
# x_151 => add_75
triton_per_fused_add_cat_native_layer_norm_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 50
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 50)
    tmp19 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x3)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 50, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((49*r2) + (25088*x1) + (((-1) + x0) % 49)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp7, tmp17)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 512, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = tmp22 - tmp32
    tmp40 = 512.0
    tmp41 = tmp38 / tmp40
    tmp42 = 1e-06
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp22, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp49, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5t5lh3qvkoae7yjgikimpwtvukpg4fxgnaqxr6vbv3tj5p2giq.py
# Source Nodes: [x4_3, x_feat], Original ATen: [aten.add, aten.native_layer_norm]
# x4_3 => add_79
# x_feat => var_mean_20
triton_per_fused_add_native_layer_norm_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 400
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
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sz/cszzkbcghr66aarocngzc6igx4t2xystsdn2gdlx44dfjjov23og.py
# Source Nodes: [x_162], Original ATen: [aten.clone]
# x_162 => clone_64
triton_poi_fused_clone_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25600*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (25600*x1)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (50*x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (50*x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 512.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 64), (64, 64, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (1, 1, 320), (320, 320, 1))
    assert_size_stride(arg19_1, (320, ), (1, ))
    assert_size_stride(arg20_1, (320, ), (1, ))
    assert_size_stride(arg21_1, (320, ), (1, ))
    assert_size_stride(arg22_1, (320, ), (1, ))
    assert_size_stride(arg23_1, (320, ), (1, ))
    assert_size_stride(arg24_1, (320, ), (1, ))
    assert_size_stride(arg25_1, (320, ), (1, ))
    assert_size_stride(arg26_1, (320, ), (1, ))
    assert_size_stride(arg27_1, (1, 1, 512), (512, 512, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (192, 64), (64, 1))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg47_1, (16, ), (1, ))
    assert_size_stride(arg48_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg49_1, (24, ), (1, ))
    assert_size_stride(arg50_1, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg51_1, (24, ), (1, ))
    assert_size_stride(arg52_1, (64, 64), (64, 1))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (512, 64), (64, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (64, 512), (512, 1))
    assert_size_stride(arg57_1, (64, ), (1, ))
    assert_size_stride(arg58_1, (192, 64), (64, 1))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (64, 64), (64, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (512, 64), (64, 1))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (64, 512), (512, 1))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (384, 128), (128, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg75_1, (32, ), (1, ))
    assert_size_stride(arg76_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg77_1, (48, ), (1, ))
    assert_size_stride(arg78_1, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg79_1, (48, ), (1, ))
    assert_size_stride(arg80_1, (128, 128), (128, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (1024, 128), (128, 1))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (128, 1024), (1024, 1))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (384, 128), (128, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (128, 128), (128, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (1024, 128), (128, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (128, 1024), (1024, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg95_1, (320, ), (1, ))
    assert_size_stride(arg96_1, (320, ), (1, ))
    assert_size_stride(arg97_1, (320, ), (1, ))
    assert_size_stride(arg98_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg99_1, (320, ), (1, ))
    assert_size_stride(arg100_1, (960, 320), (320, 1))
    assert_size_stride(arg101_1, (960, ), (1, ))
    assert_size_stride(arg102_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (80, ), (1, ))
    assert_size_stride(arg104_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (120, ), (1, ))
    assert_size_stride(arg106_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg107_1, (120, ), (1, ))
    assert_size_stride(arg108_1, (320, 320), (320, 1))
    assert_size_stride(arg109_1, (320, ), (1, ))
    assert_size_stride(arg110_1, (1280, 320), (320, 1))
    assert_size_stride(arg111_1, (1280, ), (1, ))
    assert_size_stride(arg112_1, (320, 1280), (1280, 1))
    assert_size_stride(arg113_1, (320, ), (1, ))
    assert_size_stride(arg114_1, (960, 320), (320, 1))
    assert_size_stride(arg115_1, (960, ), (1, ))
    assert_size_stride(arg116_1, (320, 320), (320, 1))
    assert_size_stride(arg117_1, (320, ), (1, ))
    assert_size_stride(arg118_1, (1280, 320), (320, 1))
    assert_size_stride(arg119_1, (1280, ), (1, ))
    assert_size_stride(arg120_1, (320, 1280), (1280, 1))
    assert_size_stride(arg121_1, (320, ), (1, ))
    assert_size_stride(arg122_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (1536, 512), (512, 1))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg135_1, (192, ), (1, ))
    assert_size_stride(arg136_1, (512, 512), (512, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (2048, 512), (512, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (512, 2048), (2048, 1))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (1536, 512), (512, 1))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (512, 512), (512, 1))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (2048, 512), (512, 1))
    assert_size_stride(arg147_1, (2048, ), (1, ))
    assert_size_stride(arg148_1, (512, 2048), (2048, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (1000, 512), (512, 1))
    assert_size_stride(arg151_1, (1000, ), (1, ))
    assert_size_stride(arg152_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg152_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg152_1
        buf1 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg38_1, buf1, 192, 16, grid=grid(192, 16), stream=stream0)
        del arg38_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cuda', dtype=torch.float32)
        # Source Nodes: [x1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf2, arg39_1, buf3, buf4, 25088, 64, grid=grid(25088), stream=stream0)
        buf6 = empty_strided((8, 3137, 64), (200768, 1, 3137), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_39], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(arg0_1, buf2, arg39_1, buf3, buf4, arg40_1, arg41_1, buf6, 1606144, grid=grid(1606144), stream=stream0)
        del arg0_1
        del arg39_1
        del arg40_1
        del arg41_1
        del buf3
        del buf4
        buf7 = reinterpret_tensor(buf2, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf2  # reuse
        # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(buf6, buf7, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg42_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf8, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del buf7
        buf12 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_38, cur], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_5.run(buf6, buf8, arg43_1, arg1_1, arg2_1, buf12, 25096, 64, grid=grid(25096), stream=stream0)
        del arg1_1
        del arg2_1
        buf13 = empty((25096, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf12, (25096, 64), (64, 1), 0), reinterpret_tensor(arg44_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf13)
        del arg44_1
        del arg45_1
        buf14 = empty_strided((8, 8, 1, 8, 25), (1600, 8, 12800, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf13, buf14, 12800, 126, grid=grid(12800), stream=stream0)
        buf15 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf14, buf15, 512, 25, grid=grid(512), stream=stream0)
        buf16 = buf14; del buf14  # reuse
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf13, buf15, buf16, 12800, 126, grid=grid(12800), stream=stream0)
        buf17 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf16, buf17, 512, 25, grid=grid(512), stream=stream0)
        buf18 = reinterpret_tensor(buf12, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf12  # reuse
        # Source Nodes: [k_softmax], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_10.run(buf13, buf15, buf17, buf18, 1606144, grid=grid(1606144), stream=stream0)
        buf19 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf13, buf19, 1606144, grid=grid(1606144), stream=stream0)
        buf20 = empty((64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf19, (64, 3137, 8), (25096, 8, 1), 0), out=buf20)
        buf21 = buf19; del buf19  # reuse
        # Source Nodes: [factor_att_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf13, buf21, 1606144, grid=grid(1606144), stream=stream0)
        buf22 = reinterpret_tensor(buf18, (64, 3137, 8), (25096, 8, 1), 0); del buf18  # reuse
        # Source Nodes: [factor_att_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf20, (64, 8, 8), (64, 8, 1), 0), out=buf22)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf23, (8, 16, 56, 56), (50176, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg48_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf24, (8, 24, 56, 56), (75264, 3136, 56, 1))
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg50_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf25, (8, 24, 56, 56), (75264, 3136, 56, 1))
        buf26 = reinterpret_tensor(buf21, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf21  # reuse
        # Source Nodes: [x_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf22, buf13, buf23, arg47_1, buf24, arg49_1, buf25, arg51_1, buf26, 1606144, grid=grid(1606144), stream=stream0)
        del buf23
        del buf24
        del buf25
        buf27 = reinterpret_tensor(buf22, (25096, 64), (64, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (25096, 64), (64, 1), 0), reinterpret_tensor(arg52_1, (64, 64), (1, 64), 0), out=buf27)
        del arg52_1
        buf28 = reinterpret_tensor(buf26, (8, 3137, 64), (200768, 1, 3137), 0); del buf26  # reuse
        buf32 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_38, cur_2, x_13], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_14.run(buf6, buf8, arg43_1, buf27, arg53_1, arg3_1, arg4_1, buf28, buf32, 25096, 64, grid=grid(25096), stream=stream0)
        del arg3_1
        del arg4_1
        del arg53_1
        del buf27
        del buf8
        buf33 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (25096, 64), (64, 1), 0), reinterpret_tensor(arg54_1, (64, 512), (1, 64), 0), out=buf33)
        del arg54_1
        buf34 = reinterpret_tensor(buf33, (8, 3137, 512), (1606144, 512, 1), 0); del buf33  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf34, arg55_1, 12849152, grid=grid(12849152), stream=stream0)
        del arg55_1
        buf35 = reinterpret_tensor(buf32, (25096, 64), (64, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (25096, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 64), (1, 512), 0), out=buf35)
        del arg56_1
        buf36 = reinterpret_tensor(buf35, (8, 3137, 64), (200768, 64, 1), 0); del buf35  # reuse
        # Source Nodes: [x1_2], Original ATen: [aten.add]
        triton_poi_fused_add_16.run(buf36, buf28, arg57_1, 25096, 64, grid=grid(25096, 64), stream=stream0)
        del arg57_1
        # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(reinterpret_tensor(buf36, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), arg42_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf37, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg42_1
        buf38 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_36, cur_4], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_17.run(buf36, buf37, arg43_1, buf38, buf39, 25096, 64, grid=grid(25096), stream=stream0)
        buf41 = reinterpret_tensor(buf28, (8, 3137, 64), (200768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [cat_36, cur_4], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_18.run(buf36, buf37, arg43_1, buf38, buf39, arg5_1, arg6_1, buf41, 512, 3137, grid=grid(512, 3137), stream=stream0)
        del arg5_1
        del arg6_1
        del buf38
        del buf39
        buf42 = buf13; del buf13  # reuse
        # Source Nodes: [l__mod___serial_blocks1_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf41, (25096, 64), (64, 1), 0), reinterpret_tensor(arg58_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf42)
        del arg58_1
        del arg59_1
        buf43 = buf16; del buf16  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_6.run(buf42, buf43, 12800, 126, grid=grid(12800), stream=stream0)
        buf44 = buf17; del buf17  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_7.run(buf43, buf44, 512, 25, grid=grid(512), stream=stream0)
        buf45 = buf43; del buf43  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_red_fused__softmax_8.run(buf42, buf44, buf45, 12800, 126, grid=grid(12800), stream=stream0)
        buf46 = buf15; del buf15  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9.run(buf45, buf46, 512, 25, grid=grid(512), stream=stream0)
        del buf45
        buf47 = reinterpret_tensor(buf41, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf41  # reuse
        # Source Nodes: [k_softmax_1], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_10.run(buf42, buf44, buf46, buf47, 1606144, grid=grid(1606144), stream=stream0)
        del buf44
        del buf46
        buf48 = reinterpret_tensor(buf6, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf6  # reuse
        # Source Nodes: [factor_att_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf42, buf48, 1606144, grid=grid(1606144), stream=stream0)
        buf49 = buf20; del buf20  # reuse
        # Source Nodes: [factor_att_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf48, (64, 3137, 8), (25096, 8, 1), 0), out=buf49)
        buf50 = buf48; del buf48  # reuse
        # Source Nodes: [factor_att_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf42, buf50, 1606144, grid=grid(1606144), stream=stream0)
        buf51 = reinterpret_tensor(buf47, (64, 3137, 8), (25096, 8, 1), 0); del buf47  # reuse
        # Source Nodes: [factor_att_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf50, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf49, (64, 8, 8), (64, 8, 1), 0), out=buf51)
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf52, (8, 16, 56, 56), (50176, 3136, 56, 1))
        del arg46_1
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg48_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf53, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg48_1
        # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg50_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf54, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg50_1
        buf55 = reinterpret_tensor(buf50, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf50  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf51, buf42, buf52, arg47_1, buf53, arg49_1, buf54, arg51_1, buf55, 1606144, grid=grid(1606144), stream=stream0)
        del arg47_1
        del arg49_1
        del arg51_1
        del buf42
        del buf52
        del buf53
        del buf54
        buf56 = reinterpret_tensor(buf51, (25096, 64), (64, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf55, (25096, 64), (64, 1), 0), reinterpret_tensor(arg60_1, (64, 64), (1, 64), 0), out=buf56)
        del arg60_1
        buf57 = reinterpret_tensor(buf56, (8, 3137, 64), (200768, 64, 1), 0); del buf56  # reuse
        buf61 = reinterpret_tensor(buf55, (8, 3137, 64), (200768, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [cat_36, cur_6, x_31], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_19.run(buf57, buf36, buf37, arg43_1, arg61_1, arg7_1, arg8_1, buf61, 25096, 64, grid=grid(25096), stream=stream0)
        del arg43_1
        del arg61_1
        del arg7_1
        del arg8_1
        del buf36
        buf62 = reinterpret_tensor(buf34, (25096, 512), (512, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (25096, 64), (64, 1), 0), reinterpret_tensor(arg62_1, (64, 512), (1, 64), 0), out=buf62)
        del arg62_1
        buf63 = reinterpret_tensor(buf62, (8, 3137, 512), (1606144, 512, 1), 0); del buf62  # reuse
        # Source Nodes: [x_34], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf63, arg63_1, 12849152, grid=grid(12849152), stream=stream0)
        del arg63_1
        buf64 = reinterpret_tensor(buf61, (25096, 64), (64, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (25096, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 64), (1, 512), 0), out=buf64)
        del arg64_1
        del buf63
        buf65 = reinterpret_tensor(buf37, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf37  # reuse
        # Source Nodes: [x1_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf57, buf64, arg65_1, buf65, 1605632, grid=grid(1605632), stream=stream0)
        del arg65_1
        del buf57
        del buf64
        buf66 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x1_nocls, x_40], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_21.run(arg66_1, buf66, 8192, 4, grid=grid(8192, 4), stream=stream0)
        del arg66_1
        # Source Nodes: [x1_nocls, x_40], Original ATen: [aten.clone, aten.convolution]
        buf67 = extern_kernels.convolution(buf65, buf66, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 128, 28, 28), (100352, 784, 28, 1))
        del buf65
        del buf66
        buf68 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        buf69 = empty_strided((8, 784, 1), (784, 1, 6272), device='cuda', dtype=torch.float32)
        # Source Nodes: [x2], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_22.run(buf67, arg67_1, buf68, buf69, 6272, 128, grid=grid(6272), stream=stream0)
        buf71 = empty_strided((8, 785, 128), (100480, 1, 785), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(arg9_1, buf67, arg67_1, buf68, buf69, arg68_1, arg69_1, buf71, 803840, grid=grid(803840), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg9_1
        del buf68
        del buf69
        buf72 = reinterpret_tensor(buf67, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf67  # reuse
        # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf71, buf72, 1024, 784, grid=grid(1024, 784), stream=stream0)
        # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf73, (8, 128, 28, 28), (100352, 784, 28, 1))
        del buf72
        buf77 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_33, cur_8], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_red_fused_cat_native_layer_norm_25.run(buf71, buf73, arg71_1, arg10_1, arg11_1, buf77, 6280, 128, grid=grid(6280), stream=stream0)
        del arg10_1
        del arg11_1
        buf78 = empty((6280, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg73_1, reinterpret_tensor(buf77, (6280, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf78)
        del arg72_1
        del arg73_1
        buf79 = empty_strided((8, 8, 1, 16, 7), (896, 16, 7168, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_26.run(buf78, buf79, 7168, 113, grid=grid(7168), stream=stream0)
        buf80 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_27.run(buf79, buf80, 1024, 7, grid=grid(1024), stream=stream0)
        buf81 = buf79; del buf79  # reuse
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_red_fused__softmax_28.run(buf78, buf80, buf81, 7168, 113, grid=grid(7168), stream=stream0)
        buf82 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_per_fused__softmax_29.run(buf81, buf82, 1024, 7, grid=grid(1024), stream=stream0)
        buf83 = reinterpret_tensor(buf77, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf77  # reuse
        # Source Nodes: [k_softmax_2], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_30.run(buf78, buf80, buf82, buf83, 803840, grid=grid(803840), stream=stream0)
        buf84 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf78, buf84, 803840, grid=grid(803840), stream=stream0)
        buf85 = empty((64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf84, (64, 785, 16), (12560, 16, 1), 0), out=buf85)
        buf86 = buf84; del buf84  # reuse
        # Source Nodes: [factor_att_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf78, buf86, 803840, grid=grid(803840), stream=stream0)
        buf87 = reinterpret_tensor(buf83, (64, 785, 16), (12560, 16, 1), 0); del buf83  # reuse
        # Source Nodes: [factor_att_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf85, (64, 16, 16), (256, 16, 1), 0), out=buf87)
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg74_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf88, (8, 32, 28, 28), (25088, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg76_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf89, (8, 48, 28, 28), (37632, 784, 28, 1))
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg78_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf90, (8, 48, 28, 28), (37632, 784, 28, 1))
        buf91 = reinterpret_tensor(buf86, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf86  # reuse
        # Source Nodes: [x_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf87, buf78, buf88, arg75_1, buf89, arg77_1, buf90, arg79_1, buf91, 803840, grid=grid(803840), stream=stream0)
        del buf88
        del buf89
        del buf90
        buf92 = reinterpret_tensor(buf87, (6280, 128), (128, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (6280, 128), (128, 1), 0), reinterpret_tensor(arg80_1, (128, 128), (1, 128), 0), out=buf92)
        del arg80_1
        buf93 = reinterpret_tensor(buf91, (8, 785, 128), (100480, 1, 785), 0); del buf91  # reuse
        buf97 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_33, cur_10, x_53], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_34.run(buf71, buf73, arg71_1, buf92, arg81_1, arg12_1, arg13_1, buf93, buf97, 6280, 128, grid=grid(6280), stream=stream0)
        del arg12_1
        del arg13_1
        del arg81_1
        del buf71
        del buf73
        buf98 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (6280, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 1024), (1, 128), 0), out=buf98)
        del arg82_1
        buf99 = reinterpret_tensor(buf98, (8, 785, 1024), (803840, 1024, 1), 0); del buf98  # reuse
        # Source Nodes: [x_56], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_35.run(buf99, arg83_1, 6430720, grid=grid(6430720), stream=stream0)
        del arg83_1
        buf100 = reinterpret_tensor(buf97, (6280, 128), (128, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg84_1, (1024, 128), (1, 1024), 0), out=buf100)
        del arg84_1
        buf101 = reinterpret_tensor(buf100, (8, 785, 128), (100480, 128, 1), 0); del buf100  # reuse
        # Source Nodes: [x2_2], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf101, buf93, arg85_1, 6280, 128, grid=grid(6280, 128), stream=stream0)
        del arg85_1
        # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(reinterpret_tensor(buf101, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf102, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg70_1
        buf103 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        buf104 = empty_strided((8, 785, 1), (785, 1, 6280), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_31, cur_12], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_37.run(buf101, buf102, arg71_1, buf103, buf104, 6280, 128, grid=grid(6280), stream=stream0)
        buf106 = reinterpret_tensor(buf93, (8, 785, 128), (100480, 128, 1), 0); del buf93  # reuse
        # Source Nodes: [cat_31, cur_12], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_38.run(buf101, buf102, arg71_1, buf103, buf104, arg14_1, arg15_1, buf106, 1024, 785, grid=grid(1024, 785), stream=stream0)
        del arg14_1
        del arg15_1
        del buf103
        del buf104
        buf107 = buf78; del buf78  # reuse
        # Source Nodes: [l__mod___serial_blocks2_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg87_1, reinterpret_tensor(buf106, (6280, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf107)
        del arg86_1
        del arg87_1
        buf108 = buf81; del buf81  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_26.run(buf107, buf108, 7168, 113, grid=grid(7168), stream=stream0)
        buf109 = buf82; del buf82  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_27.run(buf108, buf109, 1024, 7, grid=grid(1024), stream=stream0)
        buf110 = buf108; del buf108  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_red_fused__softmax_28.run(buf107, buf109, buf110, 7168, 113, grid=grid(7168), stream=stream0)
        buf111 = buf80; del buf80  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_29.run(buf110, buf111, 1024, 7, grid=grid(1024), stream=stream0)
        del buf110
        buf112 = reinterpret_tensor(buf106, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf106  # reuse
        # Source Nodes: [k_softmax_3], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_30.run(buf107, buf109, buf111, buf112, 803840, grid=grid(803840), stream=stream0)
        del buf109
        del buf111
        buf113 = reinterpret_tensor(buf92, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf92  # reuse
        # Source Nodes: [factor_att_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf107, buf113, 803840, grid=grid(803840), stream=stream0)
        buf114 = buf85; del buf85  # reuse
        # Source Nodes: [factor_att_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf112, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf113, (64, 785, 16), (12560, 16, 1), 0), out=buf114)
        buf115 = buf113; del buf113  # reuse
        # Source Nodes: [factor_att_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf107, buf115, 803840, grid=grid(803840), stream=stream0)
        buf116 = reinterpret_tensor(buf112, (64, 785, 16), (12560, 16, 1), 0); del buf112  # reuse
        # Source Nodes: [factor_att_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf114, (64, 16, 16), (256, 16, 1), 0), out=buf116)
        del buf114
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(reinterpret_tensor(buf107, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg74_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf117, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg74_1
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(reinterpret_tensor(buf107, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg76_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf118, (8, 48, 28, 28), (37632, 784, 28, 1))
        del arg76_1
        # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(reinterpret_tensor(buf107, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg78_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf119, (8, 48, 28, 28), (37632, 784, 28, 1))
        del arg78_1
        buf120 = reinterpret_tensor(buf115, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf115  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf116, buf107, buf117, arg75_1, buf118, arg77_1, buf119, arg79_1, buf120, 803840, grid=grid(803840), stream=stream0)
        del arg75_1
        del arg77_1
        del arg79_1
        del buf107
        del buf117
        del buf118
        del buf119
        buf121 = reinterpret_tensor(buf116, (6280, 128), (128, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (6280, 128), (128, 1), 0), reinterpret_tensor(arg88_1, (128, 128), (1, 128), 0), out=buf121)
        del arg88_1
        buf122 = reinterpret_tensor(buf121, (8, 785, 128), (100480, 128, 1), 0); del buf121  # reuse
        buf126 = reinterpret_tensor(buf120, (8, 785, 128), (100480, 128, 1), 0); del buf120  # reuse
        # Source Nodes: [cat_31, cur_14, x_71], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_39.run(buf122, buf101, buf102, arg71_1, arg89_1, arg16_1, arg17_1, buf126, 6280, 128, grid=grid(6280), stream=stream0)
        del arg16_1
        del arg17_1
        del arg71_1
        del arg89_1
        del buf101
        buf127 = reinterpret_tensor(buf99, (6280, 1024), (1024, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (6280, 128), (128, 1), 0), reinterpret_tensor(arg90_1, (128, 1024), (1, 128), 0), out=buf127)
        del arg90_1
        buf128 = reinterpret_tensor(buf127, (8, 785, 1024), (803840, 1024, 1), 0); del buf127  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_35.run(buf128, arg91_1, 6430720, grid=grid(6430720), stream=stream0)
        del arg91_1
        buf129 = reinterpret_tensor(buf126, (6280, 128), (128, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 128), (1, 1024), 0), out=buf129)
        del arg92_1
        del buf128
        buf130 = reinterpret_tensor(buf102, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf102  # reuse
        # Source Nodes: [x2_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf122, buf129, arg93_1, buf130, 802816, grid=grid(802816), stream=stream0)
        del arg93_1
        del buf122
        del buf129
        buf131 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x2_nocls, x_80], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_41.run(arg94_1, buf131, 40960, 4, grid=grid(40960, 4), stream=stream0)
        del arg94_1
        # Source Nodes: [x2_nocls, x_80], Original ATen: [aten.clone, aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 320, 14, 14), (62720, 196, 14, 1))
        del buf130
        del buf131
        buf133 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf134 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        buf135 = empty_strided((8, 196, 1, 3), (588, 1, 4704, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_42.run(buf132, arg95_1, buf133, buf134, buf135, 4704, 107, grid=grid(4704), stream=stream0)
        buf136 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        buf137 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_43.run(buf133, buf134, buf135, buf136, buf137, 1568, 3, grid=grid(1568), stream=stream0)
        del buf133
        del buf134
        del buf135
        buf139 = empty_strided((8, 197, 320), (63040, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_44.run(arg18_1, buf132, arg95_1, buf136, buf137, arg96_1, arg97_1, buf139, 504320, grid=grid(504320), stream=stream0)
        del arg18_1
        del arg95_1
        del arg96_1
        del arg97_1
        buf140 = reinterpret_tensor(buf132, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf132  # reuse
        # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(buf139, buf140, 2560, 196, grid=grid(2560, 196), stream=stream0)
        # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg98_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf141, (8, 320, 14, 14), (62720, 196, 14, 1))
        del buf140
        buf142 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        buf143 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        buf144 = empty_strided((8, 197, 1, 3), (591, 1, 4728, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_red_fused_cat_native_layer_norm_46.run(buf139, buf141, arg99_1, buf142, buf143, buf144, 4728, 107, grid=grid(4728), stream=stream0)
        buf145 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        buf146 = empty_strided((8, 197, 1), (197, 1, 1576), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_47.run(buf142, buf143, buf144, buf145, buf146, 1576, 3, grid=grid(1576), stream=stream0)
        buf148 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28, cur_16], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_48.run(buf139, buf141, arg99_1, buf145, buf146, arg19_1, arg20_1, buf148, 504320, grid=grid(504320), stream=stream0)
        del arg19_1
        del arg20_1
        buf149 = empty((1576, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf148, (1576, 320), (320, 1), 0), reinterpret_tensor(arg100_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf149)
        del arg100_1
        del arg101_1
        buf150 = empty_strided((8, 8, 1, 40, 2), (640, 40, 5120, 1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_49.run(buf149, buf150, 5120, 99, grid=grid(5120), stream=stream0)
        buf151 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_50.run(buf150, buf151, 2560, 2, grid=grid(2560), stream=stream0)
        buf152 = buf150; del buf150  # reuse
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_51.run(buf149, buf151, buf152, 5120, 99, grid=grid(5120), stream=stream0)
        buf153 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_per_fused__softmax_52.run(buf152, buf153, 2560, 2, grid=grid(2560), stream=stream0)
        buf154 = reinterpret_tensor(buf148, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf148  # reuse
        # Source Nodes: [k_softmax_4], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_53.run(buf149, buf151, buf153, buf154, 504320, grid=grid(504320), stream=stream0)
        buf155 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf149, buf155, 504320, grid=grid(504320), stream=stream0)
        buf156 = empty((64, 40, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf154, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf155, (64, 197, 40), (7880, 40, 1), 0), out=buf156)
        buf157 = buf155; del buf155  # reuse
        # Source Nodes: [factor_att_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_55.run(buf149, buf157, 504320, grid=grid(504320), stream=stream0)
        buf158 = reinterpret_tensor(buf154, (64, 197, 40), (7880, 40, 1), 0); del buf154  # reuse
        # Source Nodes: [factor_att_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf156, (64, 40, 40), (1600, 40, 1), 0), out=buf158)
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf159, (8, 80, 14, 14), (15680, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg104_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf160, (8, 120, 14, 14), (23520, 196, 14, 1))
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg106_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf161, (8, 120, 14, 14), (23520, 196, 14, 1))
        buf162 = reinterpret_tensor(buf157, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf157  # reuse
        # Source Nodes: [x_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_56.run(buf158, buf149, buf159, arg103_1, buf160, arg105_1, buf161, arg107_1, buf162, 504320, grid=grid(504320), stream=stream0)
        del buf159
        del buf160
        del buf161
        buf163 = reinterpret_tensor(buf158, (1576, 320), (320, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf162, (1576, 320), (320, 1), 0), reinterpret_tensor(arg108_1, (320, 320), (1, 320), 0), out=buf163)
        del arg108_1
        buf164 = reinterpret_tensor(buf162, (8, 197, 320), (63040, 1, 197), 0); del buf162  # reuse
        # Source Nodes: [cat_28, x_93], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf139, buf141, arg99_1, buf163, arg109_1, buf164, 1576, 320, grid=grid(1576, 320), stream=stream0)
        del arg109_1
        del buf141
        buf165 = buf144; del buf144  # reuse
        buf166 = buf143; del buf143  # reuse
        buf167 = buf142; del buf142  # reuse
        # Source Nodes: [cur_18], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_58.run(buf164, buf165, buf166, buf167, 4728, 107, grid=grid(4728), stream=stream0)
        buf168 = buf146; del buf146  # reuse
        buf169 = buf145; del buf145  # reuse
        # Source Nodes: [cur_18], Original ATen: [aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_47.run(buf165, buf166, buf167, buf168, buf169, 1576, 3, grid=grid(1576), stream=stream0)
        del buf165
        del buf166
        del buf167
        buf171 = reinterpret_tensor(buf163, (8, 197, 320), (63040, 320, 1), 0); del buf163  # reuse
        # Source Nodes: [cur_18], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_59.run(buf164, buf168, buf169, arg21_1, arg22_1, buf171, 1576, 320, grid=grid(1576, 320), stream=stream0)
        del arg21_1
        del arg22_1
        buf172 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (1576, 320), (320, 1), 0), reinterpret_tensor(arg110_1, (320, 1280), (1, 320), 0), out=buf172)
        del arg110_1
        buf173 = reinterpret_tensor(buf172, (8, 197, 1280), (252160, 1280, 1), 0); del buf172  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_60.run(buf173, arg111_1, 2017280, grid=grid(2017280), stream=stream0)
        del arg111_1
        buf174 = reinterpret_tensor(buf171, (1576, 320), (320, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg112_1, (1280, 320), (1, 1280), 0), out=buf174)
        del arg112_1
        buf175 = reinterpret_tensor(buf174, (8, 197, 320), (63040, 320, 1), 0); del buf174  # reuse
        # Source Nodes: [x3_2], Original ATen: [aten.add]
        triton_poi_fused_add_61.run(buf175, buf164, arg113_1, 1576, 320, grid=grid(1576, 320), stream=stream0)
        del arg113_1
        # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(reinterpret_tensor(buf175, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), arg98_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf176, (8, 320, 14, 14), (62720, 196, 14, 1))
        del arg98_1
        buf177 = buf169; del buf169  # reuse
        buf178 = buf168; del buf168  # reuse
        # Source Nodes: [cat_26, cur_20], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_62.run(buf175, buf176, arg99_1, buf177, buf178, 1576, 320, grid=grid(1576), stream=stream0)
        buf180 = reinterpret_tensor(buf164, (8, 197, 320), (63040, 320, 1), 0); del buf164  # reuse
        # Source Nodes: [cat_26, cur_20], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_63.run(buf175, buf176, arg99_1, buf177, buf178, arg23_1, arg24_1, buf180, 2560, 197, grid=grid(2560, 197), stream=stream0)
        del arg23_1
        del arg24_1
        del buf177
        del buf178
        buf181 = buf149; del buf149  # reuse
        # Source Nodes: [l__mod___serial_blocks3_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg115_1, reinterpret_tensor(buf180, (1576, 320), (320, 1), 0), reinterpret_tensor(arg114_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf181)
        del arg114_1
        del arg115_1
        buf182 = buf152; del buf152  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_49.run(buf181, buf182, 5120, 99, grid=grid(5120), stream=stream0)
        buf183 = buf153; del buf153  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_50.run(buf182, buf183, 2560, 2, grid=grid(2560), stream=stream0)
        buf184 = buf182; del buf182  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_red_fused__softmax_51.run(buf181, buf183, buf184, 5120, 99, grid=grid(5120), stream=stream0)
        buf185 = buf151; del buf151  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_per_fused__softmax_52.run(buf184, buf185, 2560, 2, grid=grid(2560), stream=stream0)
        del buf184
        buf186 = reinterpret_tensor(buf180, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf180  # reuse
        # Source Nodes: [k_softmax_5], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_53.run(buf181, buf183, buf185, buf186, 504320, grid=grid(504320), stream=stream0)
        del buf183
        del buf185
        buf187 = reinterpret_tensor(buf139, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf139  # reuse
        # Source Nodes: [factor_att_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_54.run(buf181, buf187, 504320, grid=grid(504320), stream=stream0)
        buf188 = buf156; del buf156  # reuse
        # Source Nodes: [factor_att_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf187, (64, 197, 40), (7880, 40, 1), 0), out=buf188)
        buf189 = buf187; del buf187  # reuse
        # Source Nodes: [factor_att_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_55.run(buf181, buf189, 504320, grid=grid(504320), stream=stream0)
        buf190 = reinterpret_tensor(buf186, (64, 197, 40), (7880, 40, 1), 0); del buf186  # reuse
        # Source Nodes: [factor_att_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf188, (64, 40, 40), (1600, 40, 1), 0), out=buf190)
        del buf188
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(reinterpret_tensor(buf181, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
        assert_size_stride(buf191, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg102_1
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(reinterpret_tensor(buf181, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg104_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf192, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg104_1
        # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(reinterpret_tensor(buf181, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg106_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf193, (8, 120, 14, 14), (23520, 196, 14, 1))
        del arg106_1
        buf194 = reinterpret_tensor(buf189, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf189  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_56.run(buf190, buf181, buf191, arg103_1, buf192, arg105_1, buf193, arg107_1, buf194, 504320, grid=grid(504320), stream=stream0)
        del arg103_1
        del arg105_1
        del arg107_1
        del buf181
        del buf191
        del buf192
        del buf193
        buf195 = reinterpret_tensor(buf190, (1576, 320), (320, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (1576, 320), (320, 1), 0), reinterpret_tensor(arg116_1, (320, 320), (1, 320), 0), out=buf195)
        del arg116_1
        buf196 = reinterpret_tensor(buf195, (8, 197, 320), (63040, 320, 1), 0); del buf195  # reuse
        buf200 = reinterpret_tensor(buf194, (8, 197, 320), (63040, 320, 1), 0); del buf194  # reuse
        # Source Nodes: [cat_26, cur_22, x_111], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_64.run(buf196, buf175, buf176, arg99_1, arg117_1, arg25_1, arg26_1, buf200, 1576, 320, grid=grid(1576), stream=stream0)
        del arg117_1
        del arg25_1
        del arg26_1
        del arg99_1
        del buf175
        buf201 = reinterpret_tensor(buf173, (1576, 1280), (1280, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf200, (1576, 320), (320, 1), 0), reinterpret_tensor(arg118_1, (320, 1280), (1, 320), 0), out=buf201)
        del arg118_1
        buf202 = reinterpret_tensor(buf201, (8, 197, 1280), (252160, 1280, 1), 0); del buf201  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_60.run(buf202, arg119_1, 2017280, grid=grid(2017280), stream=stream0)
        del arg119_1
        buf203 = reinterpret_tensor(buf200, (1576, 320), (320, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg120_1, (1280, 320), (1, 1280), 0), out=buf203)
        del arg120_1
        del buf202
        buf204 = reinterpret_tensor(buf176, (8, 320, 14, 14), (62720, 1, 4480, 320), 0); del buf176  # reuse
        # Source Nodes: [x3_nocls], Original ATen: [aten.clone]
        triton_poi_fused_clone_65.run(buf196, buf203, arg121_1, buf204, 501760, grid=grid(501760), stream=stream0)
        del arg121_1
        del buf196
        del buf203
        buf205 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [x3_nocls, x_120], Original ATen: [aten.clone, aten.convolution]
        triton_poi_fused_clone_convolution_66.run(arg122_1, buf205, 163840, 4, grid=grid(163840, 4), stream=stream0)
        del arg122_1
        # Source Nodes: [x3_nocls, x_120], Original ATen: [aten.clone, aten.convolution]
        buf206 = extern_kernels.convolution(buf204, buf205, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 512, 7, 7), (25088, 49, 7, 1))
        del buf204
        del buf205
        buf207 = reinterpret_tensor(buf137, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf137  # reuse
        buf208 = reinterpret_tensor(buf136, (8, 49, 1, 4), (196, 1, 1568, 49), 0); del buf136  # reuse
        buf209 = empty_strided((8, 49, 1, 4), (196, 1, 1568, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_67.run(buf206, arg123_1, buf207, buf208, buf209, 1568, 128, grid=grid(1568), stream=stream0)
        buf210 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        buf211 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [x4], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_68.run(buf207, buf208, buf209, buf210, buf211, 392, 4, grid=grid(392), stream=stream0)
        del buf207
        del buf208
        del buf209
        buf213 = empty_strided((8, 50, 512), (25600, 1, 50), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(arg27_1, buf206, arg123_1, buf210, buf211, arg124_1, arg125_1, buf213, 204800, grid=grid(204800), stream=stream0)
        del arg123_1
        del arg124_1
        del arg125_1
        del arg27_1
        del buf210
        del buf211
        buf214 = reinterpret_tensor(buf206, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf206  # reuse
        # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_70.run(buf213, buf214, 4096, 49, grid=grid(4096, 49), stream=stream0)
        # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf215, (8, 512, 7, 7), (25088, 49, 7, 1))
        del buf214
        buf216 = empty_strided((8, 50, 1, 4), (200, 1, 1600, 50), device='cuda', dtype=torch.float32)
        buf217 = empty_strided((8, 50, 1, 4), (200, 1, 1600, 50), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((8, 50, 1, 4), (200, 1, 1600, 50), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_red_fused_cat_native_layer_norm_71.run(buf213, buf215, arg127_1, buf216, buf217, buf218, 1600, 128, grid=grid(1600), stream=stream0)
        buf219 = empty_strided((8, 50, 1), (50, 1, 400), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((8, 50, 1), (50, 1, 400), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_72.run(buf216, buf217, buf218, buf219, buf220, 400, 4, grid=grid(400), stream=stream0)
        buf222 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_23, cur_24], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_73.run(buf213, buf215, arg127_1, buf219, buf220, arg28_1, arg29_1, buf222, 204800, grid=grid(204800), stream=stream0)
        del arg28_1
        del arg29_1
        buf223 = empty((400, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg129_1, reinterpret_tensor(buf222, (400, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf223)
        del arg128_1
        del arg129_1
        buf224 = reinterpret_tensor(buf49, (8, 8, 1, 64), (512, 64, 4096, 1), 0); del buf49  # reuse
        buf225 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
        triton_per_fused__softmax_74.run(buf223, buf224, buf225, 4096, 50, grid=grid(4096), stream=stream0)
        buf226 = reinterpret_tensor(buf222, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf222  # reuse
        # Source Nodes: [k_softmax_6], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_75.run(buf223, buf224, buf225, buf226, 204800, grid=grid(204800), stream=stream0)
        buf227 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_76.run(buf223, buf227, 204800, grid=grid(204800), stream=stream0)
        buf228 = empty((64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [factor_att_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf227, (64, 50, 64), (3200, 64, 1), 0), out=buf228)
        buf229 = buf227; del buf227  # reuse
        # Source Nodes: [factor_att_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_77.run(buf223, buf229, 204800, grid=grid(204800), stream=stream0)
        buf230 = reinterpret_tensor(buf226, (64, 50, 64), (3200, 64, 1), 0); del buf226  # reuse
        # Source Nodes: [factor_att_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf228, (64, 64, 64), (4096, 64, 1), 0), out=buf230)
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg130_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf231, (8, 128, 7, 7), (6272, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg132_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf232, (8, 192, 7, 7), (9408, 49, 7, 1))
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg134_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf233, (8, 192, 7, 7), (9408, 49, 7, 1))
        buf234 = reinterpret_tensor(buf229, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf230, buf223, buf231, arg131_1, buf232, arg133_1, buf233, arg135_1, buf234, 204800, grid=grid(204800), stream=stream0)
        del buf231
        del buf232
        del buf233
        buf235 = reinterpret_tensor(buf230, (400, 512), (512, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (400, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 512), (1, 512), 0), out=buf235)
        del arg136_1
        buf236 = reinterpret_tensor(buf234, (8, 50, 512), (25600, 1, 50), 0); del buf234  # reuse
        # Source Nodes: [cat_23, x_133], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_79.run(buf213, buf215, arg127_1, buf235, arg137_1, buf236, 400, 512, grid=grid(400, 512), stream=stream0)
        del arg137_1
        del buf215
        buf237 = buf218; del buf218  # reuse
        buf238 = buf217; del buf217  # reuse
        buf239 = buf216; del buf216  # reuse
        # Source Nodes: [cur_26], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_80.run(buf236, buf237, buf238, buf239, 1600, 128, grid=grid(1600), stream=stream0)
        buf240 = buf220; del buf220  # reuse
        buf241 = buf219; del buf219  # reuse
        # Source Nodes: [cur_26], Original ATen: [aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_72.run(buf237, buf238, buf239, buf240, buf241, 400, 4, grid=grid(400), stream=stream0)
        del buf237
        del buf238
        del buf239
        buf243 = reinterpret_tensor(buf235, (8, 50, 512), (25600, 512, 1), 0); del buf235  # reuse
        # Source Nodes: [cur_26], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_81.run(buf236, buf240, buf241, arg30_1, arg31_1, buf243, 400, 512, grid=grid(400, 512), stream=stream0)
        del arg30_1
        del arg31_1
        buf244 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (400, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 2048), (1, 512), 0), out=buf244)
        del arg138_1
        buf245 = reinterpret_tensor(buf244, (8, 50, 2048), (102400, 2048, 1), 0); del buf244  # reuse
        # Source Nodes: [x_136], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_82.run(buf245, arg139_1, 819200, grid=grid(819200), stream=stream0)
        del arg139_1
        buf246 = reinterpret_tensor(buf243, (400, 512), (512, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf245, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 512), (1, 2048), 0), out=buf246)
        del arg140_1
        buf247 = reinterpret_tensor(buf246, (8, 50, 512), (25600, 512, 1), 0); del buf246  # reuse
        # Source Nodes: [x4_2], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf247, buf236, arg141_1, 400, 512, grid=grid(400, 512), stream=stream0)
        del arg141_1
        # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj_1], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(reinterpret_tensor(buf247, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), arg126_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf248, (8, 512, 7, 7), (25088, 49, 7, 1))
        del arg126_1
        buf249 = buf241; del buf241  # reuse
        buf250 = buf240; del buf240  # reuse
        # Source Nodes: [cat_21, cur_28], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_84.run(buf247, buf248, arg127_1, buf249, buf250, 400, 512, grid=grid(400), stream=stream0)
        buf252 = reinterpret_tensor(buf236, (8, 50, 512), (25600, 512, 1), 0); del buf236  # reuse
        # Source Nodes: [cat_21, cur_28], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_poi_fused_cat_native_layer_norm_85.run(buf247, buf248, arg127_1, buf249, buf250, arg32_1, arg33_1, buf252, 4096, 50, grid=grid(4096, 50), stream=stream0)
        del arg32_1
        del arg33_1
        buf253 = buf223; del buf223  # reuse
        # Source Nodes: [l__mod___serial_blocks4_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg143_1, reinterpret_tensor(buf252, (400, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf253)
        del arg142_1
        del arg143_1
        buf254 = buf225; del buf225  # reuse
        buf255 = buf224; del buf224  # reuse
        # Source Nodes: [k_softmax_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_74.run(buf253, buf254, buf255, 4096, 50, grid=grid(4096), stream=stream0)
        buf256 = reinterpret_tensor(buf252, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf252  # reuse
        # Source Nodes: [k_softmax_7], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_75.run(buf253, buf254, buf255, buf256, 204800, grid=grid(204800), stream=stream0)
        del buf254
        buf257 = reinterpret_tensor(buf213, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf213  # reuse
        # Source Nodes: [factor_att_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_76.run(buf253, buf257, 204800, grid=grid(204800), stream=stream0)
        buf258 = buf228; del buf228  # reuse
        # Source Nodes: [factor_att_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf256, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf257, (64, 50, 64), (3200, 64, 1), 0), out=buf258)
        buf259 = buf257; del buf257  # reuse
        # Source Nodes: [factor_att_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_77.run(buf253, buf259, 204800, grid=grid(204800), stream=stream0)
        buf260 = reinterpret_tensor(buf256, (64, 50, 64), (3200, 64, 1), 0); del buf256  # reuse
        # Source Nodes: [factor_att_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf259, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf258, (64, 64, 64), (4096, 64, 1), 0), out=buf260)
        del buf258
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg130_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf261, (8, 128, 7, 7), (6272, 49, 7, 1))
        del arg130_1
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg132_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf262, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg132_1
        # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg134_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf263, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg134_1
        buf264 = reinterpret_tensor(buf259, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [x_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_78.run(buf260, buf253, buf261, arg131_1, buf262, arg133_1, buf263, arg135_1, buf264, 204800, grid=grid(204800), stream=stream0)
        del arg131_1
        del arg133_1
        del arg135_1
        del buf253
        del buf261
        del buf262
        del buf263
        buf265 = reinterpret_tensor(buf260, (400, 512), (512, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (400, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 512), (1, 512), 0), out=buf265)
        del arg144_1
        buf266 = reinterpret_tensor(buf265, (8, 50, 512), (25600, 512, 1), 0); del buf265  # reuse
        buf270 = reinterpret_tensor(buf264, (8, 50, 512), (25600, 512, 1), 0); del buf264  # reuse
        # Source Nodes: [cat_21, cur_30, x_151], Original ATen: [aten.add, aten.cat, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_86.run(buf266, buf247, buf248, arg127_1, arg145_1, arg34_1, arg35_1, buf270, 400, 512, grid=grid(400), stream=stream0)
        del arg127_1
        del arg145_1
        del arg34_1
        del arg35_1
        del buf247
        del buf248
        buf271 = reinterpret_tensor(buf245, (400, 2048), (2048, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf270, (400, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 2048), (1, 512), 0), out=buf271)
        del arg146_1
        buf272 = reinterpret_tensor(buf271, (8, 50, 2048), (102400, 2048, 1), 0); del buf271  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_82.run(buf272, arg147_1, 819200, grid=grid(819200), stream=stream0)
        del arg147_1
        buf273 = reinterpret_tensor(buf270, (400, 512), (512, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg148_1, (2048, 512), (1, 2048), 0), out=buf273)
        del arg148_1
        del buf272
        buf274 = buf250; del buf250  # reuse
        buf275 = buf249; del buf249  # reuse
        # Source Nodes: [x4_3, x_feat], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_87.run(buf266, buf273, arg149_1, buf274, buf275, 400, 512, grid=grid(400), stream=stream0)
        buf277 = reinterpret_tensor(buf255, (8, 512), (512, 1), 0); del buf255  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.clone]
        triton_poi_fused_clone_88.run(buf266, buf273, arg149_1, buf274, buf275, arg36_1, arg37_1, buf277, 4096, grid=grid(4096), stream=stream0)
        del arg149_1
        del arg36_1
        del arg37_1
        del buf266
        del buf273
        del buf274
        del buf275
        buf278 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162, x_163], Original ATen: [aten.addmm, aten.clone]
        extern_kernels.addmm(arg151_1, buf277, reinterpret_tensor(arg150_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf278)
        del arg150_1
        del arg151_1
        return (buf278, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1, 1, 320), (320, 320, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
