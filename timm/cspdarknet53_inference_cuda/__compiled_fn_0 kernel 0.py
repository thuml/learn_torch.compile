
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


# kernel path: /tmp/torchinductor_youkaichao/rr/crrwkxcpboofanir4ahdijl45q63dz5oiulxumeqxm5cu3ocy5kg.py
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
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbxnrzrqop4klqko3lxrugnst7kip4kb4m4tqgyf3ww2kcgb2l.py
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
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgf2cdwnxcsgc77mimqgzep6vlc4nhvlqd4nmstffcz44qlbcd3v.py
# Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_5 => gt, mul_3, where
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (32*x2) + (2097152*y1)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlu4wxexnuui7velxqflmxeh4f7nxn5ps5nrojr4ky4uvxyodat.py
# Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.leaky_relu]
# x_5 => gt, mul_3, where
# x_6 => convolution_1
triton_poi_fused_convolution_leaky_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvls2vh2mhtt6mnlx2snjzmh2427vljzta3wrd6vds3rxv3m5lbz.py
# Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_10 => gt_1, mul_7, where_1
# x_7 => add_3, mul_5, mul_6, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clipm7d3xwkxpldyerl3d5psshnthwq4zebfxi7lpvbvp67xfbop.py
# Source Nodes: [x_14, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_14 => add_5, mul_10, mul_9, sub_2
# x_18 => gt_2, mul_11, where_2
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2dmddp6k5wttkcj4cbj667d36yr3hx42mca2jd2dclpcdiugxo.py
# Source Nodes: [x_19], Original ATen: [aten.convolution]
# x_19 => convolution_3
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (1048576 + x2 + (16384*y0) + (2097152*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cumntw2ej3hjughm5mhybxd7a5jw4mg2wbzteojn4n7nvxcg4jvm.py
# Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_20 => add_7, mul_13, mul_14, sub_3
# x_24 => gt_3, mul_15, where_3
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (32*x2) + (524288*y1)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccp42xuiqd32ww4czdgn7j6j7hbtcqxkmnva27djc5uexgc5eusj.py
# Source Nodes: [x_27, x_31, xb_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# x_27 => add_9, mul_17, mul_18, sub_4
# x_31 => gt_4, mul_19, where_4
# xb_1 => add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 16384
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (1048576 + x2 + (16384*y0) + (2097152*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (y0 + (64*x2) + (1048576*y1)), tmp21, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqb627snntznrvrmwve24g2rhnmsj2e3ojn5kd2qpc3iui63pd5.py
# Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_34 => add_12, mul_21, mul_22, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7s4kdfxibzenbmi57bo5o3ukpijq7z7muu35hzgu6r4ibpbuxsj.py
# Source Nodes: [cat_9], Original ATen: [aten.cat]
# cat_9 => cat
triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 128
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 128)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (16384*y3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-1048576) + x2 + (16384*y0) + (1048576*y1)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (y0 + (128*x2) + (2097152*y1)), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnftnlzhllqq6w2hlhooisw4irtw7stv2vku2juupj7njpomton.py
# Source Nodes: [out, x_43], Original ATen: [aten.convolution, aten.leaky_relu]
# out => gt_6, mul_27, where_6
# x_43 => convolution_7
triton_poi_fused_convolution_leaky_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbjmmlfipulxgj5zaawozf33swtoj36tloqdybklszq2nwbdzmz.py
# Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_44 => add_16, mul_29, mul_30, sub_7
# x_47 => gt_7, mul_31, where_7
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpiaxx4ijpyzotkdmqdgktzk6npuh2u7xoha2iuouwdk2xfgmil.py
# Source Nodes: [x_51, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_51 => add_18, mul_33, mul_34, sub_8
# x_55 => gt_8, mul_35, where_8
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfdpdb75gmpdnbxpraziaswqqqrx5mroq365ofmcjo4o5vtzcyxu.py
# Source Nodes: [x_56], Original ATen: [aten.convolution]
# x_56 => convolution_9
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (262144 + x2 + (4096*y0) + (524288*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czuzhekarwjgcpepvuuqdd3mcmzeoosk42nxrt2zlrpesadukbb4.py
# Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_57 => add_20, mul_37, mul_38, sub_9
# x_61 => gt_9, mul_39, where_9
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznopd662quulaqcyykbzu7erzocids4tk5ohz3bprwxhyruvc6e.py
# Source Nodes: [x_61, x_63], Original ATen: [aten.convolution, aten.leaky_relu]
# x_61 => gt_9, mul_39, where_9
# x_63 => convolution_10
triton_poi_fused_convolution_leaky_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xdq25nbe6trt2ptfyccjw2utb35qshiqmikokvdlnmy3533xt4.py
# Source Nodes: [shortcut_2, x_64, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# shortcut_2 => add_23
# x_64 => add_22, mul_41, mul_42, sub_10
# x_68 => gt_10, mul_43, where_10
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (262144 + x2 + (4096*y0) + (524288*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (y0 + (64*x2) + (262144*y1)), tmp21, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5zatmy5r4g5ryusivwkdaj4fzbppev7hadbosjo2sb7dfj7nhi.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_78 => add_27, mul_49, mul_50, sub_12
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcvwsp2beikh2rzf4blufnszzw22mtsateet4zmvdsg3tlkztm6.py
# Source Nodes: [x_82, xb_4], Original ATen: [aten.add, aten.leaky_relu]
# x_82 => gt_12, mul_51, where_12
# xb_4 => add_28
triton_poi_fused_add_leaky_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5xueaioviqky47kekskbg3akywbrckepdkj52czo6skvxaxbky.py
# Source Nodes: [cat_8], Original ATen: [aten.cat]
# cat_8 => cat_1
triton_poi_fused_cat_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 128
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 128)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (4096*y3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-262144) + x2 + (4096*y0) + (262144*y1)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (y0 + (128*x2) + (524288*y1)), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwos4nwfriiwfry3qayyegkw4py27bzqkzfvzmyvh5t2aduhahba.py
# Source Nodes: [out_1, x_94], Original ATen: [aten.convolution, aten.leaky_relu]
# out_1 => gt_14, mul_59, where_14
# x_94 => convolution_15
triton_poi_fused_convolution_leaky_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curzfyx6ozyntuxpfusiyf4xka2nxoiivsu45cdbfgdr3wqmnwse.py
# Source Nodes: [x_95, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_95 => add_34, mul_61, mul_62, sub_15
# x_98 => gt_15, mul_63, where_15
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (256*x2) + (262144*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxv35dao77p36udkfyefzmbqcphfi6sp3a2blfz5vghapv37uad.py
# Source Nodes: [x_102, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_102 => add_36, mul_65, mul_66, sub_16
# x_106 => gt_16, mul_67, where_16
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2wnefyzpb56wtcxkslcsj47imgyxrzcgbuufxlklspjksvwvtua.py
# Source Nodes: [x_107], Original ATen: [aten.convolution]
# x_107 => convolution_17
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
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (131072 + x2 + (1024*y0) + (262144*y1)), xmask)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrobhcmvo726tap3dvh4cayyjej3tqp43ct7ege5nosx5e5fkge.py
# Source Nodes: [x_108, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_108 => add_38, mul_69, mul_70, sub_17
# x_112 => gt_17, mul_71, where_17
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask)
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnps2xtmpjzn7cxswjs2pnxt57c2hbql6y4os7kvuln65deiyzrr.py
# Source Nodes: [x_112, x_114], Original ATen: [aten.convolution, aten.leaky_relu]
# x_112 => gt_17, mul_71, where_17
# x_114 => convolution_18
triton_poi_fused_convolution_leaky_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3ex537p3rmdhdnpkzslcym2hvf2oh225icfqe7cdmdibp6hd6xd.py
# Source Nodes: [shortcut_4, x_115, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# shortcut_4 => add_41
# x_115 => add_40, mul_73, mul_74, sub_18
# x_119 => gt_18, mul_75, where_18
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask)
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (131072 + x2 + (1024*y0) + (262144*y1)), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (y0 + (128*x2) + (131072*y1)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3r3le4jbxhydlx7u7y2thtxrq6qf666ufshjhmnglpe6xtseb6i.py
# Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_129 => add_45, mul_81, mul_82, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fya5gyckk63llclg4gyjp2s6icelfslgffmyoke3ufdnvcrvmu.py
# Source Nodes: [shortcut_5, x_133], Original ATen: [aten.add, aten.leaky_relu]
# shortcut_5 => add_46
# x_133 => gt_20, mul_83, where_20
triton_poi_fused_add_leaky_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csk2myfdrbj2vetqycl3fq5u5qncyftbox6ptdojo47hzuzr4rqy.py
# Source Nodes: [cat_7], Original ATen: [aten.cat]
# cat_7 => cat_2
triton_poi_fused_cat_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 256
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 256)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (1024*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 256, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131072) + x2 + (1024*y0) + (131072*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (y0 + (256*x2) + (262144*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cam7t6l7tmm4pkicqsdvka3bkw6r7kgmjfy56hzdfitsbutuxof6.py
# Source Nodes: [out_2, x_229], Original ATen: [aten.convolution, aten.leaky_relu]
# out_2 => gt_34, mul_139, where_34
# x_229 => convolution_35
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73pxvb7tsmbzivte3pihk7gmmpznnnutqejloltx5q4agadhry3.py
# Source Nodes: [x_230, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_230 => add_82, mul_141, mul_142, sub_35
# x_233 => gt_35, mul_143, where_35
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (512*x2) + (131072*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tnq4say2knacoqnhjdphhgscndyv5jsk6zwzq42jvj3snqehkk.py
# Source Nodes: [x_237, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_237 => add_84, mul_145, mul_146, sub_36
# x_241 => gt_36, mul_147, where_36
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidiwvcfo76sbxnnz3rvlejw4tthrbfpn23wvrheskgzioa337sp.py
# Source Nodes: [x_242], Original ATen: [aten.convolution]
# x_242 => convolution_37
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (65536 + x2 + (256*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5phmrirsfmy7wgpzgp6rh4zrytjk7hgcfdaemalkryldlehqx3.py
# Source Nodes: [x_243, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_243 => add_86, mul_149, mul_150, sub_37
# x_247 => gt_37, mul_151, where_37
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlgtfzukhfohmauydt4tmrzvo7pjw2gbn4tzvn7ji2ldzr75qob.py
# Source Nodes: [x_247, x_249], Original ATen: [aten.convolution, aten.leaky_relu]
# x_247 => gt_37, mul_151, where_37
# x_249 => convolution_38
triton_poi_fused_convolution_leaky_relu_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzz4ffmxtselhdgmgw3263rcvltlo4ddg3752h2holk44fht7am.py
# Source Nodes: [shortcut_12, x_250, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# shortcut_12 => add_89
# x_250 => add_88, mul_153, mul_154, sub_38
# x_254 => gt_38, mul_155, where_38
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (65536 + x2 + (256*y0) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (y0 + (256*x2) + (65536*y1)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwja552mb2ghoxutow3f6yoraw7nqkvveonbauwob5enmxbp3vq.py
# Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_264 => add_93, mul_161, mul_162, sub_40
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3a24tkb5yrcqlcv7ipdr6gf6474axa24f6qoj2s4tih2vwicgv.py
# Source Nodes: [shortcut_13, x_268], Original ATen: [aten.add, aten.leaky_relu]
# shortcut_13 => add_94
# x_268 => gt_40, mul_163, where_40
triton_poi_fused_add_leaky_relu_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3n5qy7cucb5zoptbiw5qlasxgmwg3eeriwy7jgvmk6wxufpkwth.py
# Source Nodes: [cat_6], Original ATen: [aten.cat]
# cat_6 => cat_3
triton_poi_fused_cat_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 512
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 512)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (256*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 512, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-65536) + x2 + (256*y0) + (65536*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (y0 + (512*x2) + (131072*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fq/cfqhu7oaryuwm4qtg4y2abu5kjr5323xra35q2rdgbbajoxa2vo4.py
# Source Nodes: [out_3, x_364], Original ATen: [aten.convolution, aten.leaky_relu]
# out_3 => gt_54, mul_219, where_54
# x_364 => convolution_55
triton_poi_fused_convolution_leaky_relu_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7fwh4udemhnvo4msnj7ej5zdtyad2u5jo5eufsstdsrtx4gcfk.py
# Source Nodes: [x_365, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_365 => add_130, mul_221, mul_222, sub_55
# x_368 => gt_55, mul_223, where_55
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (1024*x2) + (65536*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covzwdq5vjkxxmrrwzl3wzxb2smhngun5rvz4yok62y2vl5ggyyf.py
# Source Nodes: [x_372, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_372 => add_132, mul_225, mul_226, sub_56
# x_376 => gt_56, mul_227, where_56
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizm5jhk6mecz3ycz3moej2vei5hmk3m5vzpfuus32injsm2qmok.py
# Source Nodes: [x_377], Original ATen: [aten.convolution]
# x_377 => convolution_57
triton_poi_fused_convolution_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (32768 + x2 + (64*y0) + (65536*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cc/cccid3c6cs6mh6lvanp7qwugzthzl65v2f4lgocye6mcxqdopria.py
# Source Nodes: [x_378, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# x_378 => add_134, mul_229, mul_230, sub_57
# x_382 => gt_57, mul_231, where_57
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/cloupvq75f4yk3qug52v5ayvei3dkz7ln5z3rt432sch4hhc7ige.py
# Source Nodes: [x_382, x_384], Original ATen: [aten.convolution, aten.leaky_relu]
# x_382 => gt_57, mul_231, where_57
# x_384 => convolution_58
triton_poi_fused_convolution_leaky_relu_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbckevmze4cea6yzcjjqcejukradvtypejtudgdnlxoh5yvjlyy.py
# Source Nodes: [shortcut_20, x_385, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# shortcut_20 => add_137
# x_385 => add_136, mul_233, mul_234, sub_58
# x_389 => gt_58, mul_235, where_58
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (32768 + x2 + (64*y0) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp21 = tmp19 + tmp20
    tl.store(out_ptr0 + (y0 + (512*x2) + (32768*y1)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckx36otcstb3o34flj553cnyt37sywdmzepv2tbrxqop4t5cgxb7.py
# Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_399 => add_141, mul_241, mul_242, sub_60
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6o6fpe4q3qzt42jz444cy6y6hlchhdhcn5kjmyjcnqgudv43rcq.py
# Source Nodes: [shortcut_21, x_403], Original ATen: [aten.add, aten.leaky_relu]
# shortcut_21 => add_142
# x_403 => gt_60, mul_243, where_60
triton_poi_fused_add_leaky_relu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp6 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp7, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqevd265s3duxsqlb6l43jo5hv3yddkyl7j5pzbxrfxdnmuud5uf.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat_4
triton_poi_fused_cat_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1024
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 1024)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1024, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-32768) + x2 + (64*y0) + (32768*y1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.where(tmp4, tmp7, tmp18)
    tl.store(out_ptr0 + (y0 + (1024*x2) + (65536*y1)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cof3bnletwheqsmtgbyzyerqj4iqolnfm6uzvv7fu72ywfnqobyi.py
# Source Nodes: [x_439, x_444, x_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.mean]
# x_439 => add_156, mul_265, mul_266, sub_66
# x_444 => gt_66, mul_267, where_66
# x_445 => mean
triton_per_fused__native_batch_norm_legit_no_training_leaky_relu_mean_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_leaky_relu_mean_51', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 64.0
    tmp25 = tmp23 / tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp25, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, ), (1, ))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg135_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg136_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg137_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg138_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg139_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg140_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg141_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg142_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg143_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg144_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg145_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg146_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg147_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg148_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg149_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg150_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg151_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg152_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg153_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg154_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg155_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg156_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg157_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg158_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg159_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg160_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg161_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg162_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg163_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg164_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg165_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg166_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg167_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg168_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg169_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg170_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg171_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg172_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg173_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg174_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg175_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg176_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg177_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg178_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg179_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg180_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg181_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg182_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg183_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg184_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg185_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg186_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg187_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg188_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg189_1, (1024, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg190_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg191_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg193_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg194_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg195_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg196_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg197_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg198_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg199_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg200_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg201_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg202_1, (1000, ), (1, ))
    assert_size_stride(arg203_1, (32, ), (1, ))
    assert_size_stride(arg204_1, (32, ), (1, ))
    assert_size_stride(arg205_1, (64, ), (1, ))
    assert_size_stride(arg206_1, (64, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (128, ), (1, ))
    assert_size_stride(arg209_1, (32, ), (1, ))
    assert_size_stride(arg210_1, (32, ), (1, ))
    assert_size_stride(arg211_1, (64, ), (1, ))
    assert_size_stride(arg212_1, (64, ), (1, ))
    assert_size_stride(arg213_1, (64, ), (1, ))
    assert_size_stride(arg214_1, (64, ), (1, ))
    assert_size_stride(arg215_1, (64, ), (1, ))
    assert_size_stride(arg216_1, (64, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (128, ), (1, ))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (64, ), (1, ))
    assert_size_stride(arg222_1, (64, ), (1, ))
    assert_size_stride(arg223_1, (64, ), (1, ))
    assert_size_stride(arg224_1, (64, ), (1, ))
    assert_size_stride(arg225_1, (64, ), (1, ))
    assert_size_stride(arg226_1, (64, ), (1, ))
    assert_size_stride(arg227_1, (64, ), (1, ))
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (64, ), (1, ))
    assert_size_stride(arg230_1, (64, ), (1, ))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (128, ), (1, ))
    assert_size_stride(arg241_1, (128, ), (1, ))
    assert_size_stride(arg242_1, (128, ), (1, ))
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, ), (1, ))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, ), (1, ))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, ), (1, ))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (128, ), (1, ))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (128, ), (1, ))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (512, ), (1, ))
    assert_size_stride(arg277_1, (256, ), (1, ))
    assert_size_stride(arg278_1, (256, ), (1, ))
    assert_size_stride(arg279_1, (256, ), (1, ))
    assert_size_stride(arg280_1, (256, ), (1, ))
    assert_size_stride(arg281_1, (256, ), (1, ))
    assert_size_stride(arg282_1, (256, ), (1, ))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (256, ), (1, ))
    assert_size_stride(arg285_1, (256, ), (1, ))
    assert_size_stride(arg286_1, (256, ), (1, ))
    assert_size_stride(arg287_1, (256, ), (1, ))
    assert_size_stride(arg288_1, (256, ), (1, ))
    assert_size_stride(arg289_1, (256, ), (1, ))
    assert_size_stride(arg290_1, (256, ), (1, ))
    assert_size_stride(arg291_1, (256, ), (1, ))
    assert_size_stride(arg292_1, (256, ), (1, ))
    assert_size_stride(arg293_1, (256, ), (1, ))
    assert_size_stride(arg294_1, (256, ), (1, ))
    assert_size_stride(arg295_1, (256, ), (1, ))
    assert_size_stride(arg296_1, (256, ), (1, ))
    assert_size_stride(arg297_1, (256, ), (1, ))
    assert_size_stride(arg298_1, (256, ), (1, ))
    assert_size_stride(arg299_1, (256, ), (1, ))
    assert_size_stride(arg300_1, (256, ), (1, ))
    assert_size_stride(arg301_1, (256, ), (1, ))
    assert_size_stride(arg302_1, (256, ), (1, ))
    assert_size_stride(arg303_1, (256, ), (1, ))
    assert_size_stride(arg304_1, (256, ), (1, ))
    assert_size_stride(arg305_1, (256, ), (1, ))
    assert_size_stride(arg306_1, (256, ), (1, ))
    assert_size_stride(arg307_1, (256, ), (1, ))
    assert_size_stride(arg308_1, (256, ), (1, ))
    assert_size_stride(arg309_1, (256, ), (1, ))
    assert_size_stride(arg310_1, (256, ), (1, ))
    assert_size_stride(arg311_1, (512, ), (1, ))
    assert_size_stride(arg312_1, (512, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1024, ), (1, ))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (512, ), (1, ))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (512, ), (1, ))
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (512, ), (1, ))
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, ), (1, ))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg337_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg337_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg134_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg134_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 256, 256), (2097152, 65536, 256, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 32, 256, 256), (2097152, 1, 8192, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_2.run(buf3, arg203_1, arg204_1, arg0_1, arg1_1, buf4, 256, 65536, grid=grid(256, 65536), stream=stream0)
        del arg0_1
        del arg1_1
        del arg203_1
        del arg204_1
        del buf3
        buf5 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_3.run(arg135_1, buf5, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg135_1
        # Source Nodes: [x_5, x_6], Original ATen: [aten.convolution, aten.leaky_relu]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf7, arg205_1, arg206_1, arg2_1, arg3_1, buf8, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg205_1
        del arg206_1
        del arg2_1
        del arg3_1
        del buf7
        # Source Nodes: [x_10, x_13], Original ATen: [aten.convolution, aten.leaky_relu]
        buf9 = extern_kernels.convolution(buf8, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        del arg136_1
        buf10 = buf9; del buf9  # reuse
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [x_14, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_5.run(buf11, arg207_1, arg208_1, arg4_1, arg5_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg207_1
        del arg208_1
        del arg4_1
        del arg5_1
        buf12 = buf8; del buf8  # reuse
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf11, buf12, 512, 16384, grid=grid(512, 16384), stream=stream0)
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del arg137_1
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_7.run(buf14, arg209_1, arg210_1, arg6_1, arg7_1, buf15, 256, 16384, grid=grid(256, 16384), stream=stream0)
        del arg209_1
        del arg210_1
        del arg6_1
        del arg7_1
        del buf14
        buf16 = buf5; del buf5  # reuse
        # Source Nodes: [x_24, x_26], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_3.run(arg138_1, buf16, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg138_1
        # Source Nodes: [x_24, x_26], Original ATen: [aten.convolution, aten.leaky_relu]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del buf16
        buf18 = buf17; del buf17  # reuse
        buf19 = buf12; del buf12  # reuse
        # Source Nodes: [x_27, x_31, xb_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_8.run(buf18, arg211_1, arg212_1, arg8_1, arg9_1, buf11, buf19, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg211_1
        del arg212_1
        del arg8_1
        del arg9_1
        del buf18
        # Source Nodes: [x_31, x_33, xb_1], Original ATen: [aten.add, aten.convolution, aten.leaky_relu]
        buf20 = extern_kernels.convolution(buf19, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg139_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf21, arg213_1, arg214_1, arg10_1, arg11_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg11_1
        del arg213_1
        del arg214_1
        buf22 = reinterpret_tensor(buf4, (8, 128, 128, 128), (2097152, 1, 16384, 128), 0); del buf4  # reuse
        # Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf11, buf21, buf22, 1024, 16384, grid=grid(1024, 16384), stream=stream0)
        del buf11
        # Source Nodes: [cat_9, x_38], Original ATen: [aten.cat, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg140_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = reinterpret_tensor(buf21, (8, 64, 128, 128), (1048576, 1, 8192, 64), 0); del buf21  # reuse
        # Source Nodes: [out, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_4.run(buf24, arg215_1, arg216_1, arg12_1, arg13_1, buf25, 512, 16384, grid=grid(512, 16384), stream=stream0)
        del arg12_1
        del arg13_1
        del arg215_1
        del arg216_1
        del buf24
        buf26 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out, x_43], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_11.run(arg141_1, buf26, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg141_1
        # Source Nodes: [out, x_43], Original ATen: [aten.convolution, aten.leaky_relu]
        buf27 = extern_kernels.convolution(buf25, buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf25
        del buf26
        buf28 = buf27; del buf27  # reuse
        buf29 = reinterpret_tensor(buf15, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf15  # reuse
        # Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12.run(buf28, arg217_1, arg218_1, arg14_1, arg15_1, buf29, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del arg14_1
        del arg15_1
        del arg217_1
        del arg218_1
        del buf28
        # Source Nodes: [x_47, x_50], Original ATen: [aten.convolution, aten.leaky_relu]
        buf30 = extern_kernels.convolution(buf29, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg142_1
        buf31 = buf30; del buf30  # reuse
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [x_51, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_13.run(buf32, arg219_1, arg220_1, arg16_1, arg17_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg16_1
        del arg17_1
        del arg219_1
        del arg220_1
        buf33 = empty_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf32, buf33, 512, 4096, grid=grid(512, 4096), stream=stream0)
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg143_1
        buf35 = buf34; del buf34  # reuse
        buf36 = buf33; del buf33  # reuse
        # Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15.run(buf35, arg221_1, arg222_1, arg18_1, arg19_1, buf36, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg18_1
        del arg19_1
        del arg221_1
        del arg222_1
        del buf35
        buf37 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_63], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_16.run(arg144_1, buf37, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg144_1
        # Source Nodes: [x_61, x_63], Original ATen: [aten.convolution, aten.leaky_relu]
        buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf39 = buf38; del buf38  # reuse
        buf40 = buf36; del buf36  # reuse
        # Source Nodes: [shortcut_2, x_64, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_17.run(buf39, arg223_1, arg224_1, arg20_1, arg21_1, buf32, buf40, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg20_1
        del arg21_1
        del arg223_1
        del arg224_1
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg145_1
        buf42 = buf41; del buf41  # reuse
        buf43 = reinterpret_tensor(buf39, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf39  # reuse
        # Source Nodes: [x_71, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_15.run(buf42, arg225_1, arg226_1, arg22_1, arg23_1, buf43, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del arg225_1
        del arg226_1
        del arg22_1
        del arg23_1
        del buf42
        buf44 = buf37; del buf37  # reuse
        # Source Nodes: [x_75, x_77], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_16.run(arg146_1, buf44, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg146_1
        # Source Nodes: [x_75, x_77], Original ATen: [aten.convolution, aten.leaky_relu]
        buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del buf43
        del buf44
        buf46 = buf45; del buf45  # reuse
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf46, arg227_1, arg228_1, arg24_1, arg25_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg227_1
        del arg228_1
        del arg24_1
        del arg25_1
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [x_82, xb_4], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_19.run(buf47, buf46, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del buf46
        # Source Nodes: [x_82, x_84, xb_4], Original ATen: [aten.add, aten.convolution, aten.leaky_relu]
        buf48 = extern_kernels.convolution(buf47, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg147_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [x_85], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf49, arg229_1, arg230_1, arg26_1, arg27_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg229_1
        del arg230_1
        del arg26_1
        del arg27_1
        buf50 = buf29; del buf29  # reuse
        # Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf32, buf49, buf50, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf32
        # Source Nodes: [cat_8, x_89], Original ATen: [aten.cat, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg148_1
        buf52 = buf51; del buf51  # reuse
        buf53 = buf50; del buf50  # reuse
        # Source Nodes: [out_1, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_12.run(buf52, arg231_1, arg232_1, arg28_1, arg29_1, buf53, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del arg231_1
        del arg232_1
        del arg28_1
        del arg29_1
        del buf52
        buf54 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, x_94], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_21.run(arg149_1, buf54, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg149_1
        # Source Nodes: [out_1, x_94], Original ATen: [aten.convolution, aten.leaky_relu]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf53
        del buf54
        buf56 = buf55; del buf55  # reuse
        buf57 = reinterpret_tensor(buf49, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf49  # reuse
        # Source Nodes: [x_95, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22.run(buf56, arg233_1, arg234_1, arg30_1, arg31_1, buf57, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del arg233_1
        del arg234_1
        del arg30_1
        del arg31_1
        del buf56
        # Source Nodes: [x_101, x_98], Original ATen: [aten.convolution, aten.leaky_relu]
        buf58 = extern_kernels.convolution(buf57, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg150_1
        buf59 = buf58; del buf58  # reuse
        buf60 = buf59; del buf59  # reuse
        # Source Nodes: [x_102, x_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_23.run(buf60, arg235_1, arg236_1, arg32_1, arg33_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg235_1
        del arg236_1
        del arg32_1
        del arg33_1
        buf61 = empty_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(buf60, buf61, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        # Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg151_1
        buf63 = buf62; del buf62  # reuse
        buf64 = buf61; del buf61  # reuse
        # Source Nodes: [x_108, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf63, arg237_1, arg238_1, arg34_1, arg35_1, buf64, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg237_1
        del arg238_1
        del arg34_1
        del arg35_1
        del buf63
        buf65 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112, x_114], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg152_1, buf65, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg152_1
        # Source Nodes: [x_112, x_114], Original ATen: [aten.convolution, aten.leaky_relu]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf67 = buf66; del buf66  # reuse
        buf68 = buf64; del buf64  # reuse
        # Source Nodes: [shortcut_4, x_115, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_27.run(buf67, arg239_1, arg240_1, arg36_1, arg37_1, buf60, buf68, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg239_1
        del arg240_1
        del arg36_1
        del arg37_1
        # Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg153_1
        buf70 = buf69; del buf69  # reuse
        buf71 = reinterpret_tensor(buf67, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf67  # reuse
        # Source Nodes: [x_122, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf70, arg241_1, arg242_1, arg38_1, arg39_1, buf71, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg241_1
        del arg242_1
        del arg38_1
        del arg39_1
        del buf70
        buf72 = buf65; del buf65  # reuse
        # Source Nodes: [x_126, x_128], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg154_1, buf72, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg154_1
        # Source Nodes: [x_126, x_128], Original ATen: [aten.convolution, aten.leaky_relu]
        buf73 = extern_kernels.convolution(buf71, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf71
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_129], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf74, arg243_1, arg244_1, arg40_1, arg41_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg243_1
        del arg244_1
        del arg40_1
        del arg41_1
        buf75 = buf68; del buf68  # reuse
        # Source Nodes: [shortcut_5, x_133], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf75, buf74, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg155_1
        buf77 = buf76; del buf76  # reuse
        buf78 = reinterpret_tensor(buf74, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf74  # reuse
        # Source Nodes: [x_136, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf77, arg245_1, arg246_1, arg42_1, arg43_1, buf78, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg245_1
        del arg246_1
        del arg42_1
        del arg43_1
        del buf77
        buf79 = buf72; del buf72  # reuse
        # Source Nodes: [x_140, x_142], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg156_1, buf79, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg156_1
        # Source Nodes: [x_140, x_142], Original ATen: [aten.convolution, aten.leaky_relu]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf78
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf81, arg247_1, arg248_1, arg44_1, arg45_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg247_1
        del arg248_1
        del arg44_1
        del arg45_1
        buf82 = buf75; del buf75  # reuse
        # Source Nodes: [shortcut_6, x_147], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf82, buf81, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg157_1
        buf84 = buf83; del buf83  # reuse
        buf85 = reinterpret_tensor(buf81, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf81  # reuse
        # Source Nodes: [x_150, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf84, arg249_1, arg250_1, arg46_1, arg47_1, buf85, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg249_1
        del arg250_1
        del arg46_1
        del arg47_1
        del buf84
        buf86 = buf79; del buf79  # reuse
        # Source Nodes: [x_154, x_156], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg158_1, buf86, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg158_1
        # Source Nodes: [x_154, x_156], Original ATen: [aten.convolution, aten.leaky_relu]
        buf87 = extern_kernels.convolution(buf85, buf86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf85
        buf88 = buf87; del buf87  # reuse
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf88, arg251_1, arg252_1, arg48_1, arg49_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg251_1
        del arg252_1
        del arg48_1
        del arg49_1
        buf89 = buf82; del buf82  # reuse
        # Source Nodes: [shortcut_7, x_161], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf89, buf88, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg159_1
        buf91 = buf90; del buf90  # reuse
        buf92 = reinterpret_tensor(buf88, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf88  # reuse
        # Source Nodes: [x_164, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf91, arg253_1, arg254_1, arg50_1, arg51_1, buf92, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg253_1
        del arg254_1
        del arg50_1
        del arg51_1
        del buf91
        buf93 = buf86; del buf86  # reuse
        # Source Nodes: [x_168, x_170], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg160_1, buf93, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg160_1
        # Source Nodes: [x_168, x_170], Original ATen: [aten.convolution, aten.leaky_relu]
        buf94 = extern_kernels.convolution(buf92, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf92
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf95, arg255_1, arg256_1, arg52_1, arg53_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg255_1
        del arg256_1
        del arg52_1
        del arg53_1
        buf96 = buf89; del buf89  # reuse
        # Source Nodes: [shortcut_8, x_175], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf96, buf95, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg161_1
        buf98 = buf97; del buf97  # reuse
        buf99 = reinterpret_tensor(buf95, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf95  # reuse
        # Source Nodes: [x_178, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf98, arg257_1, arg258_1, arg54_1, arg55_1, buf99, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg257_1
        del arg258_1
        del arg54_1
        del arg55_1
        del buf98
        buf100 = buf93; del buf93  # reuse
        # Source Nodes: [x_182, x_184], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg162_1, buf100, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg162_1
        # Source Nodes: [x_182, x_184], Original ATen: [aten.convolution, aten.leaky_relu]
        buf101 = extern_kernels.convolution(buf99, buf100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf99
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf102, arg259_1, arg260_1, arg56_1, arg57_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg259_1
        del arg260_1
        del arg56_1
        del arg57_1
        buf103 = buf96; del buf96  # reuse
        # Source Nodes: [shortcut_9, x_189], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf103, buf102, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg163_1
        buf105 = buf104; del buf104  # reuse
        buf106 = reinterpret_tensor(buf102, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf102  # reuse
        # Source Nodes: [x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf105, arg261_1, arg262_1, arg58_1, arg59_1, buf106, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg261_1
        del arg262_1
        del arg58_1
        del arg59_1
        del buf105
        buf107 = buf100; del buf100  # reuse
        # Source Nodes: [x_196, x_198], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg164_1, buf107, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg164_1
        # Source Nodes: [x_196, x_198], Original ATen: [aten.convolution, aten.leaky_relu]
        buf108 = extern_kernels.convolution(buf106, buf107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf106
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf109, arg263_1, arg264_1, arg60_1, arg61_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg263_1
        del arg264_1
        del arg60_1
        del arg61_1
        buf110 = buf103; del buf103  # reuse
        # Source Nodes: [shortcut_10, x_203], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf110, buf109, 8192, 128, grid=grid(8192, 128), stream=stream0)
        # Source Nodes: [x_205], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg165_1
        buf112 = buf111; del buf111  # reuse
        buf113 = reinterpret_tensor(buf109, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf109  # reuse
        # Source Nodes: [x_206, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_25.run(buf112, arg265_1, arg266_1, arg62_1, arg63_1, buf113, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del arg265_1
        del arg266_1
        del arg62_1
        del arg63_1
        del buf112
        buf114 = buf107; del buf107  # reuse
        # Source Nodes: [x_210, x_212], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_26.run(arg166_1, buf114, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg166_1
        # Source Nodes: [x_210, x_212], Original ATen: [aten.convolution, aten.leaky_relu]
        buf115 = extern_kernels.convolution(buf113, buf114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del buf113
        del buf114
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [x_213], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf116, arg267_1, arg268_1, arg64_1, arg65_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg267_1
        del arg268_1
        del arg64_1
        del arg65_1
        buf117 = buf110; del buf110  # reuse
        # Source Nodes: [x_217, xb_7], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_29.run(buf117, buf116, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del buf116
        # Source Nodes: [x_217, x_219, xb_7], Original ATen: [aten.add, aten.convolution, aten.leaky_relu]
        buf118 = extern_kernels.convolution(buf117, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg167_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf119, arg269_1, arg270_1, arg66_1, arg67_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg269_1
        del arg270_1
        del arg66_1
        del arg67_1
        buf120 = buf57; del buf57  # reuse
        # Source Nodes: [cat_7], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf60, buf119, buf120, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del buf60
        # Source Nodes: [cat_7, x_224], Original ATen: [aten.cat, aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg168_1
        buf122 = buf121; del buf121  # reuse
        buf123 = buf120; del buf120  # reuse
        # Source Nodes: [out_2, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_22.run(buf122, arg271_1, arg272_1, arg68_1, arg69_1, buf123, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del arg271_1
        del arg272_1
        del arg68_1
        del arg69_1
        del buf122
        buf124 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_2, x_229], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_31.run(arg169_1, buf124, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg169_1
        # Source Nodes: [out_2, x_229], Original ATen: [aten.convolution, aten.leaky_relu]
        buf125 = extern_kernels.convolution(buf123, buf124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 512, 16, 16), (131072, 256, 16, 1))
        del buf123
        del buf124
        buf126 = buf125; del buf125  # reuse
        buf127 = reinterpret_tensor(buf119, (8, 512, 16, 16), (131072, 1, 8192, 512), 0); del buf119  # reuse
        # Source Nodes: [x_230, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32.run(buf126, arg273_1, arg274_1, arg70_1, arg71_1, buf127, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del arg273_1
        del arg274_1
        del arg70_1
        del arg71_1
        del buf126
        # Source Nodes: [x_233, x_236], Original ATen: [aten.convolution, aten.leaky_relu]
        buf128 = extern_kernels.convolution(buf127, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg170_1
        buf129 = buf128; del buf128  # reuse
        buf130 = buf129; del buf129  # reuse
        # Source Nodes: [x_237, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_33.run(buf130, arg275_1, arg276_1, arg72_1, arg73_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg275_1
        del arg276_1
        del arg72_1
        del arg73_1
        buf131 = empty_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf130, buf131, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg171_1
        buf133 = buf132; del buf132  # reuse
        buf134 = buf131; del buf131  # reuse
        # Source Nodes: [x_243, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf133, arg277_1, arg278_1, arg74_1, arg75_1, buf134, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg277_1
        del arg278_1
        del arg74_1
        del arg75_1
        del buf133
        buf135 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247, x_249], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg172_1, buf135, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg172_1
        # Source Nodes: [x_247, x_249], Original ATen: [aten.convolution, aten.leaky_relu]
        buf136 = extern_kernels.convolution(buf134, buf135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf137 = buf136; del buf136  # reuse
        buf138 = buf134; del buf134  # reuse
        # Source Nodes: [shortcut_12, x_250, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_37.run(buf137, arg279_1, arg280_1, arg76_1, arg77_1, buf130, buf138, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg279_1
        del arg280_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_256], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg173_1
        buf140 = buf139; del buf139  # reuse
        buf141 = reinterpret_tensor(buf137, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf137  # reuse
        # Source Nodes: [x_257, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf140, arg281_1, arg282_1, arg78_1, arg79_1, buf141, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg281_1
        del arg282_1
        del arg78_1
        del arg79_1
        del buf140
        buf142 = buf135; del buf135  # reuse
        # Source Nodes: [x_261, x_263], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg174_1, buf142, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg174_1
        # Source Nodes: [x_261, x_263], Original ATen: [aten.convolution, aten.leaky_relu]
        buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf141
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf144, arg283_1, arg284_1, arg80_1, arg81_1, 524288, grid=grid(524288), stream=stream0)
        del arg283_1
        del arg284_1
        del arg80_1
        del arg81_1
        buf145 = buf138; del buf138  # reuse
        # Source Nodes: [shortcut_13, x_268], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf145, buf144, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg175_1
        buf147 = buf146; del buf146  # reuse
        buf148 = reinterpret_tensor(buf144, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf144  # reuse
        # Source Nodes: [x_271, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf147, arg285_1, arg286_1, arg82_1, arg83_1, buf148, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg285_1
        del arg286_1
        del arg82_1
        del arg83_1
        del buf147
        buf149 = buf142; del buf142  # reuse
        # Source Nodes: [x_275, x_277], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg176_1, buf149, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg176_1
        # Source Nodes: [x_275, x_277], Original ATen: [aten.convolution, aten.leaky_relu]
        buf150 = extern_kernels.convolution(buf148, buf149, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf148
        buf151 = buf150; del buf150  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf151, arg287_1, arg288_1, arg84_1, arg85_1, 524288, grid=grid(524288), stream=stream0)
        del arg287_1
        del arg288_1
        del arg84_1
        del arg85_1
        buf152 = buf145; del buf145  # reuse
        # Source Nodes: [shortcut_14, x_282], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf152, buf151, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_284], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg177_1
        buf154 = buf153; del buf153  # reuse
        buf155 = reinterpret_tensor(buf151, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf151  # reuse
        # Source Nodes: [x_285, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf154, arg289_1, arg290_1, arg86_1, arg87_1, buf155, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg289_1
        del arg290_1
        del arg86_1
        del arg87_1
        del buf154
        buf156 = buf149; del buf149  # reuse
        # Source Nodes: [x_289, x_291], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg178_1, buf156, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg178_1
        # Source Nodes: [x_289, x_291], Original ATen: [aten.convolution, aten.leaky_relu]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf155
        buf158 = buf157; del buf157  # reuse
        # Source Nodes: [x_292], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf158, arg291_1, arg292_1, arg88_1, arg89_1, 524288, grid=grid(524288), stream=stream0)
        del arg291_1
        del arg292_1
        del arg88_1
        del arg89_1
        buf159 = buf152; del buf152  # reuse
        # Source Nodes: [shortcut_15, x_296], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf159, buf158, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_298], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg179_1
        buf161 = buf160; del buf160  # reuse
        buf162 = reinterpret_tensor(buf158, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf158  # reuse
        # Source Nodes: [x_299, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf161, arg293_1, arg294_1, arg90_1, arg91_1, buf162, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg293_1
        del arg294_1
        del arg90_1
        del arg91_1
        del buf161
        buf163 = buf156; del buf156  # reuse
        # Source Nodes: [x_303, x_305], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg180_1, buf163, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg180_1
        # Source Nodes: [x_303, x_305], Original ATen: [aten.convolution, aten.leaky_relu]
        buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf162
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [x_306], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf165, arg295_1, arg296_1, arg92_1, arg93_1, 524288, grid=grid(524288), stream=stream0)
        del arg295_1
        del arg296_1
        del arg92_1
        del arg93_1
        buf166 = buf159; del buf159  # reuse
        # Source Nodes: [shortcut_16, x_310], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf166, buf165, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_312], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg181_1
        buf168 = buf167; del buf167  # reuse
        buf169 = reinterpret_tensor(buf165, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf165  # reuse
        # Source Nodes: [x_313, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf168, arg297_1, arg298_1, arg94_1, arg95_1, buf169, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg297_1
        del arg298_1
        del arg94_1
        del arg95_1
        del buf168
        buf170 = buf163; del buf163  # reuse
        # Source Nodes: [x_317, x_319], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg182_1, buf170, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg182_1
        # Source Nodes: [x_317, x_319], Original ATen: [aten.convolution, aten.leaky_relu]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf169
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf172, arg299_1, arg300_1, arg96_1, arg97_1, 524288, grid=grid(524288), stream=stream0)
        del arg299_1
        del arg300_1
        del arg96_1
        del arg97_1
        buf173 = buf166; del buf166  # reuse
        # Source Nodes: [shortcut_17, x_324], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf173, buf172, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg183_1
        buf175 = buf174; del buf174  # reuse
        buf176 = reinterpret_tensor(buf172, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf172  # reuse
        # Source Nodes: [x_327, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf175, arg301_1, arg302_1, arg98_1, arg99_1, buf176, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg301_1
        del arg302_1
        del arg98_1
        del arg99_1
        del buf175
        buf177 = buf170; del buf170  # reuse
        # Source Nodes: [x_331, x_333], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg184_1, buf177, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg184_1
        # Source Nodes: [x_331, x_333], Original ATen: [aten.convolution, aten.leaky_relu]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf176
        buf179 = buf178; del buf178  # reuse
        # Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf179, arg303_1, arg304_1, arg100_1, arg101_1, 524288, grid=grid(524288), stream=stream0)
        del arg100_1
        del arg101_1
        del arg303_1
        del arg304_1
        buf180 = buf173; del buf173  # reuse
        # Source Nodes: [shortcut_18, x_338], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf180, buf179, 2048, 256, grid=grid(2048, 256), stream=stream0)
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg185_1
        buf182 = buf181; del buf181  # reuse
        buf183 = reinterpret_tensor(buf179, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf179  # reuse
        # Source Nodes: [x_341, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_35.run(buf182, arg305_1, arg306_1, arg102_1, arg103_1, buf183, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del arg102_1
        del arg103_1
        del arg305_1
        del arg306_1
        del buf182
        buf184 = buf177; del buf177  # reuse
        # Source Nodes: [x_345, x_347], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_36.run(arg186_1, buf184, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg186_1
        # Source Nodes: [x_345, x_347], Original ATen: [aten.convolution, aten.leaky_relu]
        buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 256, 16, 16), (65536, 256, 16, 1))
        del buf183
        del buf184
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [x_348], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf186, arg307_1, arg308_1, arg104_1, arg105_1, 524288, grid=grid(524288), stream=stream0)
        del arg104_1
        del arg105_1
        del arg307_1
        del arg308_1
        buf187 = buf180; del buf180  # reuse
        # Source Nodes: [x_352, xb_10], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_39.run(buf187, buf186, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf186
        # Source Nodes: [x_352, x_354, xb_10], Original ATen: [aten.add, aten.convolution, aten.leaky_relu]
        buf188 = extern_kernels.convolution(buf187, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg187_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf189, arg309_1, arg310_1, arg106_1, arg107_1, 524288, grid=grid(524288), stream=stream0)
        del arg106_1
        del arg107_1
        del arg309_1
        del arg310_1
        buf190 = buf127; del buf127  # reuse
        # Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf130, buf189, buf190, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del buf130
        # Source Nodes: [cat_6, x_359], Original ATen: [aten.cat, aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg188_1
        buf192 = buf191; del buf191  # reuse
        buf193 = buf190; del buf190  # reuse
        # Source Nodes: [out_3, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_32.run(buf192, arg311_1, arg312_1, arg108_1, arg109_1, buf193, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del arg108_1
        del arg109_1
        del arg311_1
        del arg312_1
        del buf192
        buf194 = empty_strided((1024, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_3, x_364], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_41.run(arg189_1, buf194, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg189_1
        # Source Nodes: [out_3, x_364], Original ATen: [aten.convolution, aten.leaky_relu]
        buf195 = extern_kernels.convolution(buf193, buf194, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del buf193
        del buf194
        buf196 = buf195; del buf195  # reuse
        buf197 = reinterpret_tensor(buf189, (8, 1024, 8, 8), (65536, 1, 8192, 1024), 0); del buf189  # reuse
        # Source Nodes: [x_365, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_42.run(buf196, arg313_1, arg314_1, arg110_1, arg111_1, buf197, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del arg110_1
        del arg111_1
        del arg313_1
        del arg314_1
        del buf196
        # Source Nodes: [x_368, x_371], Original ATen: [aten.convolution, aten.leaky_relu]
        buf198 = extern_kernels.convolution(buf197, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg190_1
        buf199 = buf198; del buf198  # reuse
        buf200 = buf199; del buf199  # reuse
        # Source Nodes: [x_372, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_43.run(buf200, arg315_1, arg316_1, arg112_1, arg113_1, 524288, grid=grid(524288), stream=stream0)
        del arg112_1
        del arg113_1
        del arg315_1
        del arg316_1
        buf201 = empty_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_377], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf200, buf201, 4096, 64, grid=grid(4096, 64), stream=stream0)
        # Source Nodes: [x_377], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg191_1
        buf203 = buf202; del buf202  # reuse
        buf204 = buf201; del buf201  # reuse
        # Source Nodes: [x_378, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf203, arg317_1, arg318_1, arg114_1, arg115_1, buf204, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg114_1
        del arg115_1
        del arg317_1
        del arg318_1
        del buf203
        buf205 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_382, x_384], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_46.run(arg192_1, buf205, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg192_1
        # Source Nodes: [x_382, x_384], Original ATen: [aten.convolution, aten.leaky_relu]
        buf206 = extern_kernels.convolution(buf204, buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf207 = buf206; del buf206  # reuse
        buf208 = buf204; del buf204  # reuse
        # Source Nodes: [shortcut_20, x_385, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_47.run(buf207, arg319_1, arg320_1, arg116_1, arg117_1, buf200, buf208, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg116_1
        del arg117_1
        del arg319_1
        del arg320_1
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg193_1
        buf210 = buf209; del buf209  # reuse
        buf211 = reinterpret_tensor(buf207, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf207  # reuse
        # Source Nodes: [x_392, x_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf210, arg321_1, arg322_1, arg118_1, arg119_1, buf211, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg118_1
        del arg119_1
        del arg321_1
        del arg322_1
        del buf210
        buf212 = buf205; del buf205  # reuse
        # Source Nodes: [x_396, x_398], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_46.run(arg194_1, buf212, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg194_1
        # Source Nodes: [x_396, x_398], Original ATen: [aten.convolution, aten.leaky_relu]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 512, 8, 8), (32768, 64, 8, 1))
        del buf211
        buf214 = buf213; del buf213  # reuse
        # Source Nodes: [x_399], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf214, arg323_1, arg324_1, arg120_1, arg121_1, 262144, grid=grid(262144), stream=stream0)
        del arg120_1
        del arg121_1
        del arg323_1
        del arg324_1
        buf215 = buf208; del buf208  # reuse
        # Source Nodes: [shortcut_21, x_403], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_49.run(buf215, buf214, 512, 512, grid=grid(512, 512), stream=stream0)
        # Source Nodes: [x_405], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg195_1
        buf217 = buf216; del buf216  # reuse
        buf218 = reinterpret_tensor(buf214, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf214  # reuse
        # Source Nodes: [x_406, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf217, arg325_1, arg326_1, arg122_1, arg123_1, buf218, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg122_1
        del arg123_1
        del arg325_1
        del arg326_1
        del buf217
        buf219 = buf212; del buf212  # reuse
        # Source Nodes: [x_410, x_412], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_46.run(arg196_1, buf219, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg196_1
        # Source Nodes: [x_410, x_412], Original ATen: [aten.convolution, aten.leaky_relu]
        buf220 = extern_kernels.convolution(buf218, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 512, 8, 8), (32768, 64, 8, 1))
        del buf218
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [x_413], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf221, arg327_1, arg328_1, arg124_1, arg125_1, 262144, grid=grid(262144), stream=stream0)
        del arg124_1
        del arg125_1
        del arg327_1
        del arg328_1
        buf222 = buf215; del buf215  # reuse
        # Source Nodes: [shortcut_22, x_417], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_49.run(buf222, buf221, 512, 512, grid=grid(512, 512), stream=stream0)
        # Source Nodes: [x_419], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg197_1
        buf224 = buf223; del buf223  # reuse
        buf225 = reinterpret_tensor(buf221, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf221  # reuse
        # Source Nodes: [x_420, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_45.run(buf224, arg329_1, arg330_1, arg126_1, arg127_1, buf225, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del arg126_1
        del arg127_1
        del arg329_1
        del arg330_1
        del buf224
        buf226 = buf219; del buf219  # reuse
        # Source Nodes: [x_424, x_426], Original ATen: [aten.convolution, aten.leaky_relu]
        triton_poi_fused_convolution_leaky_relu_46.run(arg198_1, buf226, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg198_1
        # Source Nodes: [x_424, x_426], Original ATen: [aten.convolution, aten.leaky_relu]
        buf227 = extern_kernels.convolution(buf225, buf226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 512, 8, 8), (32768, 64, 8, 1))
        del buf225
        del buf226
        buf228 = buf227; del buf227  # reuse
        # Source Nodes: [x_427], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf228, arg331_1, arg332_1, arg128_1, arg129_1, 262144, grid=grid(262144), stream=stream0)
        del arg128_1
        del arg129_1
        del arg331_1
        del arg332_1
        buf229 = buf222; del buf222  # reuse
        # Source Nodes: [x_431, xb_13], Original ATen: [aten.add, aten.leaky_relu]
        triton_poi_fused_add_leaky_relu_49.run(buf229, buf228, 512, 512, grid=grid(512, 512), stream=stream0)
        del buf228
        # Source Nodes: [x_431, x_433, xb_13], Original ATen: [aten.add, aten.convolution, aten.leaky_relu]
        buf230 = extern_kernels.convolution(buf229, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg199_1
        del buf229
        buf231 = buf230; del buf230  # reuse
        # Source Nodes: [x_434], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf231, arg333_1, arg334_1, arg130_1, arg131_1, 262144, grid=grid(262144), stream=stream0)
        del arg130_1
        del arg131_1
        del arg333_1
        del arg334_1
        buf232 = buf197; del buf197  # reuse
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_50.run(buf200, buf231, buf232, 8192, 64, grid=grid(8192, 64), stream=stream0)
        del buf200
        del buf231
        # Source Nodes: [cat_5, x_438], Original ATen: [aten.cat, aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg200_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf235, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf235  # reuse
        # Source Nodes: [x_439, x_444, x_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_leaky_relu_mean_51.run(buf234, buf236, arg335_1, arg336_1, arg132_1, arg133_1, 8192, 64, grid=grid(8192), stream=stream0)
        del arg132_1
        del arg133_1
        del arg335_1
        del arg336_1
        del buf234
        buf237 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_449], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg202_1, reinterpret_tensor(buf236, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf237)
        del arg201_1
        del arg202_1
        return (buf237, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('cspdarknet53', benchmark_compiled_module)
