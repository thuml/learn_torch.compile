
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
# Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
# l__mod___base_layer_0 => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/ll/cllj3w55okgo67x7wuv6n746po63pwcotjhhikcbqshipy5zqvl7.py
# Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
# l__mod___base_layer_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curp2yhe4zqtza3ba4uffwvf3o46n3jq5koe4a4xrvza6mtq7xmp.py
# Source Nodes: [l__mod___base_layer_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___base_layer_1 => add_1, mul_1, mul_2, sub
# x => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (16*x2) + (802816*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/cteqbmr6phlhvgxstl7edchw77aj4d4owvwt35po2cuj47xf4wtf.py
# Source Nodes: [l__mod___base_layer_1, l__mod___level0_0, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___base_layer_1 => add_1, mul_1, mul_2, sub
# l__mod___level0_0 => convolution_1
# x => relu
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4woqlxi3z4huxnvymznpxc6pcju5eh5br5mo6shxgl62hkzq26.py
# Source Nodes: [l__mod___level0_1, l__mod___level1_0, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___level0_1 => add_3, mul_4, mul_5, sub_1
# l__mod___level1_0 => convolution_2
# x_1 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4plaz34wwke5tpur4yg4jgmcb3vqjui6iqaftik4r4rwfhyylcr.py
# Source Nodes: [l__mod___level1_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___level1_1 => add_5, mul_7, mul_8, sub_2
# x_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3w6jheh5bhihwipramqjdbxkyq3vahp4jprqi7o3berxj5oz72x.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_1 => add_9, mul_13, mul_14, sub_4
# out_2 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnopwdorijotbnljotkuwk4ouyj42jljnogdisqsqi6rtlxvehnh.py
# Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_1 => add_9, mul_13, mul_14, sub_4
# out_2 => relu_3
# out_3 => convolution_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2ctgamxzp4bujesatizfpuggvfjigec7w7hpct6rqyvv5n57yv.py
# Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_4 => add_11, mul_16, mul_17, sub_5
# out_5 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2r/c2ruhubdy7fhixajru7ffasxochbgtxnmfj5idysznwm3v2vqm66.py
# Source Nodes: [bottom], Original ATen: [aten.max_pool2d_with_indices]
# bottom => max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 56
    x2 = (xindex // 1792)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x1) + (7168*x2)), None)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + (64*x1) + (7168*x2)), None)
    tmp3 = tl.load(in_ptr0 + (3584 + x0 + (64*x1) + (7168*x2)), None)
    tmp5 = tl.load(in_ptr0 + (3616 + x0 + (64*x1) + (7168*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjrwglztpxsew3dh6hb4hjiufai66pzivwel6apfuvc3tfa3qfnh.py
# Source Nodes: [out_7, out_8, shortcut, shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_7 => add_13, mul_19, mul_20, sub_6
# out_8 => add_14
# shortcut => add_7, mul_10, mul_11, sub_3
# shortcut_1 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7wufxf47xzubzxiiipms4cjr7lo4gojxstynl6nsw56tpeicii.py
# Source Nodes: [cat_27, out_17, out_18, x2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_27 => cat
# out_17 => add_20, mul_28, mul_29, sub_9
# out_18 => add_21
# x2 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2owk6uxnver2vclmi7kbbkntkawdmzm4pl55wakib53oxygpk7.py
# Source Nodes: [x_4, x_5, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_4 => add_23, mul_31, mul_32, sub_10
# x_5 => add_24
# x_8 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zr/czrkzn6ow7pr2imvxkajyrofetqams7z5agyvxgg6373u3y3dldf.py
# Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_21 => add_28, mul_37, mul_38, sub_12
# out_22 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjt5tol34moxhjcqwysnqixlz67cbqx7jrgwidslubkh5tebemb.py
# Source Nodes: [out_21, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_21 => add_28, mul_37, mul_38, sub_12
# out_22 => relu_10
# out_23 => convolution_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6mzzp5t5gg6gtgpdxui35goic4kip4yszq2jjuutjmrd3r4om6.py
# Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_24 => add_30, mul_40, mul_41, sub_13
# out_25 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (128*x2) + (100352*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicvon4cwvsmucioz2fv72hsxga6k4mmhofyb6yeq5ivwefweery.py
# Source Nodes: [bottom_3], Original ATen: [aten.max_pool2d_with_indices]
# bottom_3 => max_pool2d_with_indices_3
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxai6dlix4u7jqbwusvmtbb4s6643w2rfixmud4p6kcayc5lo5ce.py
# Source Nodes: [out_27, out_28, shortcut_4, shortcut_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_27 => add_32, mul_43, mul_44, sub_14
# out_28 => add_33
# shortcut_4 => add_26, mul_34, mul_35, sub_11
# shortcut_5 => relu_12
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhylgnfgzwlk2f7jgu56jyk2elu6aic6b4v2mhl3kahwzcjjitb.py
# Source Nodes: [cat_26, out_37, out_38, x2_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_26 => cat_1
# out_37 => add_39, mul_52, mul_53, sub_17
# out_38 => add_40
# x2_1 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (512*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbvlxxuipdemnubagvj2o3fbkdbpntp46kl4c2o64mhwkar75w3.py
# Source Nodes: [x1_2, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x1_2 => relu_16
# x_10 => add_42, mul_55, mul_56, sub_18
# x_11 => add_43
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgw5wakrafu2p2abwgtvr5mjkvtfnbfomynxprsrjieczekukhd.py
# Source Nodes: [cat_25, out_47, out_48, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_25 => cat_2
# out_47 => add_49, mul_64, mul_65, sub_21
# out_48 => add_50
# shortcut_7 => relu_19
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (768*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7e2yclo2wd4qlgtyi4dfzb7bowooveamgfkin75eeo3owwlcvik.py
# Source Nodes: [cat_25, out_57, out_58, x2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_25 => cat_2
# out_57 => add_56, mul_73, mul_74, sub_24
# out_58 => add_57
# x2_2 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (768*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calt7m3dptvbgjfpyxrrurivddizljka3ratl7ljx7n475vr7goq.py
# Source Nodes: [x1_4, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x1_4 => relu_23
# x_15 => add_59, mul_76, mul_77, sub_25
# x_16 => add_60
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctjpgguiq4whtuitkl5h4ay23yhrdd7s23hmjyrprz5rekcyvapr.py
# Source Nodes: [cat_23, out_67, out_68, shortcut_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_23 => cat_4
# out_67 => add_66, mul_85, mul_86, sub_28
# out_68 => add_67
# shortcut_10 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1152*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5hfxtxttybkep6mvhhpvam2t5setowexn2h7s5mhpzzpykwos5.py
# Source Nodes: [cat_23, out_97, out_98, x2_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_23 => cat_4
# out_97 => add_90, mul_115, mul_116, sub_38
# out_98 => add_91
# x2_4 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (1152*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1152*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ylz67jk3txsgnjpfkppsznrvuc5wn7prbzce6tutg2plpxkr35.py
# Source Nodes: [bottom_1], Original ATen: [aten.max_pool2d_with_indices]
# bottom_1 => max_pool2d_with_indices_1
triton_poi_fused_max_pool2d_with_indices_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + (1152*x3)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4gnzc22ks2v6azecdbs3ou2e3f4mvqqnka2ln7iekwgxakf34w.py
# Source Nodes: [x_26, x_27, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_26 => add_93, mul_118, mul_119, sub_39
# x_27 => add_94
# x_32 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1152*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3o6sbpd3lxat6b2sjq3mjbdu5kxh6ui64r6z7yfazbl6oeterrn.py
# Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_101 => add_98, mul_124, mul_125, sub_41
# out_102 => relu_38
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5agpk6pc3ktvdwn2gtxhtubomsdmu5ymxgj7n3vq7s2r73b7gw6.py
# Source Nodes: [out_101, out_102, out_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_101 => add_98, mul_124, mul_125, sub_41
# out_102 => relu_38
# out_103 => convolution_42
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6w5v7zmyic4rfzl2tktajpiee3lwrlzkwfhioiipy4x3o5fjims.py
# Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_104 => add_100, mul_127, mul_128, sub_42
# out_105 => relu_39
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (256*x2) + (50176*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/coziij6nooytppug6kddxqcohxepdmvd5p54gfmf3myf5zwfopys.py
# Source Nodes: [bottom_11], Original ATen: [aten.max_pool2d_with_indices]
# bottom_11 => max_pool2d_with_indices_7
triton_poi_fused_max_pool2d_with_indices_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/53/c53b7gqppuduyavvsaw2p57n7izgtnemgobwzllppumbknojpxuz.py
# Source Nodes: [out_107, out_108, shortcut_16, shortcut_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_107 => add_102, mul_130, mul_131, sub_43
# out_108 => add_103
# shortcut_16 => add_96, mul_121, mul_122, sub_40
# shortcut_17 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnq6yo5y4e6dpuulfjgu7ikzu43byvl5orrtois2ndgvgx2c4hxw.py
# Source Nodes: [cat_22, out_117, out_118, x2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_22 => cat_5
# out_117 => add_109, mul_139, mul_140, sub_46
# out_118 => add_110
# x2_5 => relu_43
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1024*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqr6krpgybbb3povsu3drqxemdstkrerhliwlvv7ggh3sfqvzge.py
# Source Nodes: [x1_9, x_34, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x1_9 => relu_44
# x_34 => add_112, mul_142, mul_143, sub_47
# x_35 => add_113
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvpkkakesicwqys2om735l4bebmr4wxgzw5m3hxklaxkgkisebs7.py
# Source Nodes: [cat_21, out_127, out_128, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_21 => cat_6
# out_127 => add_119, mul_151, mul_152, sub_50
# out_128 => add_120
# shortcut_19 => relu_47
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1536*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cyb6so7oceloztd6sdwk2sr73a7clf5rufzie32pof4obn44h772.py
# Source Nodes: [cat_21, out_137, out_138, x2_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_21 => cat_6
# out_137 => add_126, mul_160, mul_161, sub_53
# out_138 => add_127
# x2_6 => relu_50
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (1536*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1536*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cos5fcwpbsqsk5iuulsf7ohftlgpa746bplg2rej2kgp7hnzm55j.py
# Source Nodes: [x1_11, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x1_11 => relu_51
# x_39 => add_129, mul_163, mul_164, sub_54
# x_40 => add_130
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vsxrik7yx4cg4foqtmobtowsis5y374f7kcprul2gbtz3gj35c.py
# Source Nodes: [cat_19, out_147, out_148, shortcut_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_19 => cat_8
# out_147 => add_136, mul_172, mul_173, sub_57
# out_148 => add_137
# shortcut_22 => relu_54
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2048*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7f/c7fqreixr7x3pxbxushc3x72qhcvsr5cz2rck2qrcvwwh36qlw5d.py
# Source Nodes: [cat_19, out_177, out_178, x2_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_19 => cat_8
# out_177 => add_160, mul_202, mul_203, sub_67
# out_178 => add_161
# x2_8 => relu_64
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2048*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxku5dcep6sxvpjfdith7trhwbxnaazsc3p3a3mokmqluopzrsf.py
# Source Nodes: [cat_15, x1_15, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# x1_15 => relu_65
# x_50 => add_163, mul_205, mul_206, sub_68
# x_51 => add_164
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (2048*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2816*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co64i6dqq3y6axya2aphglyr3rxysasstdh3rp2hl6j4wgxyqvvt.py
# Source Nodes: [out_187, out_188, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_187 => add_170, mul_214, mul_215, sub_71
# out_188 => add_171
# shortcut_28 => relu_68
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xa/cxaxndcueuz6r2c7pg7afa7y5c2c5vf43a233n2rwtbjaewpblh2.py
# Source Nodes: [cat_15, out_227, out_228, shortcut_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# out_227 => add_204, mul_256, mul_257, sub_85
# out_228 => add_205
# shortcut_33 => relu_82
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2816*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklybpifxya2odqrkv7o2q2cnpgnwoez2d2sr7wfx4ggqz5sgsup.py
# Source Nodes: [cat_15, out_257, out_258, x2_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_15 => cat_12
# out_257 => add_228, mul_286, mul_287, sub_95
# out_258 => add_229
# x2_12 => relu_92
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (2816*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2816*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5s5cpjzxca45kyjrcom5xgmvpqeiy7fsf7v6pbpwovo4rps67s.py
# Source Nodes: [bottom_8], Original ATen: [aten.max_pool2d_with_indices]
# bottom_8 => max_pool2d_with_indices_4
triton_poi_fused_max_pool2d_with_indices_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + (2816*x3)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csm7fcmwedykxnxbrq3hntccztydkskjdlbgtuswx3pad33qqu5u.py
# Source Nodes: [x_73, x_74, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# x_73 => add_231, mul_289, mul_290, sub_96
# x_74 => add_232
# x_80 => relu_93
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (2816*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnikim7q7l5asmpjouaiqssmrghvygss7tdpf6anutqj3rpqbg3.py
# Source Nodes: [bottom_23, cat_14], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# bottom_23 => max_pool2d_with_indices_8
# cat_14 => cat_13
triton_poi_fused_cat_max_pool2d_with_indices_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (1024*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + (1024*x1) + (14336*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x0 + (2560*x4)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/as/casnt4ded3ysi65ov6iypufgiq4d3p2lqgqzyze6gpmgv3oe7tco.py
# Source Nodes: [out_261, out_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_261 => add_236, mul_295, mul_296, sub_98
# out_262 => relu_94
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czg2cyt3oopol32ylph27q4v55k3ydgs67e5ygzuvpmai6qu5t2u.py
# Source Nodes: [out_261, out_262, out_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# out_261 => add_236, mul_295, mul_296, sub_98
# out_262 => relu_94
# out_263 => convolution_99
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/55/c55h5jtns2lfzxawoxypbwcfchahcafbdbjppmynjiqlisvo4nqa.py
# Source Nodes: [out_264, out_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_264 => add_238, mul_298, mul_299, sub_99
# out_265 => relu_95
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (512*x2) + (25088*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cst4hblqrf7g4ujeq2jc5n6ncqr2andswbtkkvh7vgsreb7w2ixe.py
# Source Nodes: [out_267, out_268, shortcut_36, shortcut_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# out_267 => add_240, mul_301, mul_302, sub_100
# out_268 => add_241
# shortcut_36 => add_234, mul_292, mul_293, sub_97
# shortcut_37 => relu_96
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciiypvsk566wnep4ddak4wfyrsvbkgxzchluhdqgevtpk7uvzk7w.py
# Source Nodes: [cat_14, out_277, out_278, x2_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
# cat_14 => cat_13
# out_277 => add_247, mul_310, mul_311, sub_103
# out_278 => add_248
# x2_13 => relu_99
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x2 + (2560*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (2560*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4pywi4xqjinvkfcsm4z54c6mnpnqthwqo7wwltwociqffcd7qr.py
# Source Nodes: [x_82, x_83, x_87, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# x_82 => add_250, mul_313, mul_314, sub_104
# x_83 => add_251
# x_87 => relu_100
# x_88 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0 + (2560*r2) + (125440*x1)), rmask, other=0.0)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyzw6vgwhrsjfa3nhxmpl6sogeilz6hi2pshrygy3jdoril3c2a.py
# Source Nodes: [x_82, x_83, x_87, x_88, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.mean, aten.relu, aten.view]
# x_82 => add_250, mul_313, mul_314, sub_104
# x_83 => add_251
# x_87 => relu_100
# x_88 => mean
# x_92 => convolution_105
# x_93 => view
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_view_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_view_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (256, ), (1, ))
    assert_size_stride(arg78_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, ), (1, ))
    assert_size_stride(arg96_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg115_1, (256, ), (1, ))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (256, ), (1, ))
    assert_size_stride(arg138_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg187_1, (256, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (256, ), (1, ))
    assert_size_stride(arg198_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg208_1, (256, ), (1, ))
    assert_size_stride(arg209_1, (256, ), (1, ))
    assert_size_stride(arg210_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg220_1, (256, ), (1, ))
    assert_size_stride(arg221_1, (256, ), (1, ))
    assert_size_stride(arg222_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg235_1, (512, ), (1, ))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (256, ), (1, ))
    assert_size_stride(arg240_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg250_1, (256, ), (1, ))
    assert_size_stride(arg251_1, (256, ), (1, ))
    assert_size_stride(arg252_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (256, ), (1, ))
    assert_size_stride(arg255_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (256, ), (1, ))
    assert_size_stride(arg273_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg274_1, (256, ), (1, ))
    assert_size_stride(arg275_1, (256, ), (1, ))
    assert_size_stride(arg276_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg277_1, (512, ), (1, ))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg280_1, (256, ), (1, ))
    assert_size_stride(arg281_1, (256, ), (1, ))
    assert_size_stride(arg282_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg283_1, (256, ), (1, ))
    assert_size_stride(arg284_1, (256, ), (1, ))
    assert_size_stride(arg285_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (512, 2816, 1, 1), (2816, 1, 1, 1))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg307_1, (512, ), (1, ))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (1000, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg316_1, (1000, ), (1, ))
    assert_size_stride(arg317_1, (16, ), (1, ))
    assert_size_stride(arg318_1, (16, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (16, ), (1, ))
    assert_size_stride(arg321_1, (16, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (32, ), (1, ))
    assert_size_stride(arg324_1, (32, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (128, ), (1, ))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (64, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (64, ), (1, ))
    assert_size_stride(arg333_1, (64, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (128, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (64, ), (1, ))
    assert_size_stride(arg339_1, (64, ), (1, ))
    assert_size_stride(arg340_1, (), ())
    assert_size_stride(arg341_1, (64, ), (1, ))
    assert_size_stride(arg342_1, (64, ), (1, ))
    assert_size_stride(arg343_1, (), ())
    assert_size_stride(arg344_1, (128, ), (1, ))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (), ())
    assert_size_stride(arg347_1, (128, ), (1, ))
    assert_size_stride(arg348_1, (128, ), (1, ))
    assert_size_stride(arg349_1, (), ())
    assert_size_stride(arg350_1, (256, ), (1, ))
    assert_size_stride(arg351_1, (256, ), (1, ))
    assert_size_stride(arg352_1, (), ())
    assert_size_stride(arg353_1, (128, ), (1, ))
    assert_size_stride(arg354_1, (128, ), (1, ))
    assert_size_stride(arg355_1, (), ())
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (), ())
    assert_size_stride(arg359_1, (256, ), (1, ))
    assert_size_stride(arg360_1, (256, ), (1, ))
    assert_size_stride(arg361_1, (), ())
    assert_size_stride(arg362_1, (128, ), (1, ))
    assert_size_stride(arg363_1, (128, ), (1, ))
    assert_size_stride(arg364_1, (), ())
    assert_size_stride(arg365_1, (128, ), (1, ))
    assert_size_stride(arg366_1, (128, ), (1, ))
    assert_size_stride(arg367_1, (), ())
    assert_size_stride(arg368_1, (256, ), (1, ))
    assert_size_stride(arg369_1, (256, ), (1, ))
    assert_size_stride(arg370_1, (), ())
    assert_size_stride(arg371_1, (256, ), (1, ))
    assert_size_stride(arg372_1, (256, ), (1, ))
    assert_size_stride(arg373_1, (), ())
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (), ())
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (), ())
    assert_size_stride(arg380_1, (256, ), (1, ))
    assert_size_stride(arg381_1, (256, ), (1, ))
    assert_size_stride(arg382_1, (), ())
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (128, ), (1, ))
    assert_size_stride(arg385_1, (), ())
    assert_size_stride(arg386_1, (128, ), (1, ))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (), ())
    assert_size_stride(arg389_1, (256, ), (1, ))
    assert_size_stride(arg390_1, (256, ), (1, ))
    assert_size_stride(arg391_1, (), ())
    assert_size_stride(arg392_1, (256, ), (1, ))
    assert_size_stride(arg393_1, (256, ), (1, ))
    assert_size_stride(arg394_1, (), ())
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (), ())
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (), ())
    assert_size_stride(arg401_1, (256, ), (1, ))
    assert_size_stride(arg402_1, (256, ), (1, ))
    assert_size_stride(arg403_1, (), ())
    assert_size_stride(arg404_1, (128, ), (1, ))
    assert_size_stride(arg405_1, (128, ), (1, ))
    assert_size_stride(arg406_1, (), ())
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (128, ), (1, ))
    assert_size_stride(arg409_1, (), ())
    assert_size_stride(arg410_1, (256, ), (1, ))
    assert_size_stride(arg411_1, (256, ), (1, ))
    assert_size_stride(arg412_1, (), ())
    assert_size_stride(arg413_1, (256, ), (1, ))
    assert_size_stride(arg414_1, (256, ), (1, ))
    assert_size_stride(arg415_1, (), ())
    assert_size_stride(arg416_1, (128, ), (1, ))
    assert_size_stride(arg417_1, (128, ), (1, ))
    assert_size_stride(arg418_1, (), ())
    assert_size_stride(arg419_1, (128, ), (1, ))
    assert_size_stride(arg420_1, (128, ), (1, ))
    assert_size_stride(arg421_1, (), ())
    assert_size_stride(arg422_1, (256, ), (1, ))
    assert_size_stride(arg423_1, (256, ), (1, ))
    assert_size_stride(arg424_1, (), ())
    assert_size_stride(arg425_1, (128, ), (1, ))
    assert_size_stride(arg426_1, (128, ), (1, ))
    assert_size_stride(arg427_1, (), ())
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (), ())
    assert_size_stride(arg431_1, (256, ), (1, ))
    assert_size_stride(arg432_1, (256, ), (1, ))
    assert_size_stride(arg433_1, (), ())
    assert_size_stride(arg434_1, (256, ), (1, ))
    assert_size_stride(arg435_1, (256, ), (1, ))
    assert_size_stride(arg436_1, (), ())
    assert_size_stride(arg437_1, (512, ), (1, ))
    assert_size_stride(arg438_1, (512, ), (1, ))
    assert_size_stride(arg439_1, (), ())
    assert_size_stride(arg440_1, (256, ), (1, ))
    assert_size_stride(arg441_1, (256, ), (1, ))
    assert_size_stride(arg442_1, (), ())
    assert_size_stride(arg443_1, (256, ), (1, ))
    assert_size_stride(arg444_1, (256, ), (1, ))
    assert_size_stride(arg445_1, (), ())
    assert_size_stride(arg446_1, (512, ), (1, ))
    assert_size_stride(arg447_1, (512, ), (1, ))
    assert_size_stride(arg448_1, (), ())
    assert_size_stride(arg449_1, (256, ), (1, ))
    assert_size_stride(arg450_1, (256, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (256, ), (1, ))
    assert_size_stride(arg453_1, (256, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (512, ), (1, ))
    assert_size_stride(arg456_1, (512, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (512, ), (1, ))
    assert_size_stride(arg459_1, (512, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (256, ), (1, ))
    assert_size_stride(arg462_1, (256, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (256, ), (1, ))
    assert_size_stride(arg465_1, (256, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (512, ), (1, ))
    assert_size_stride(arg468_1, (512, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (256, ), (1, ))
    assert_size_stride(arg471_1, (256, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (256, ), (1, ))
    assert_size_stride(arg474_1, (256, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (512, ), (1, ))
    assert_size_stride(arg477_1, (512, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (256, ), (1, ))
    assert_size_stride(arg483_1, (256, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (256, ), (1, ))
    assert_size_stride(arg486_1, (256, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (512, ), (1, ))
    assert_size_stride(arg489_1, (512, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (256, ), (1, ))
    assert_size_stride(arg492_1, (256, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (256, ), (1, ))
    assert_size_stride(arg495_1, (256, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (512, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (512, ), (1, ))
    assert_size_stride(arg501_1, (512, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (256, ), (1, ))
    assert_size_stride(arg504_1, (256, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (256, ), (1, ))
    assert_size_stride(arg507_1, (256, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (256, ), (1, ))
    assert_size_stride(arg513_1, (256, ), (1, ))
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (256, ), (1, ))
    assert_size_stride(arg516_1, (256, ), (1, ))
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (512, ), (1, ))
    assert_size_stride(arg519_1, (512, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (512, ), (1, ))
    assert_size_stride(arg522_1, (512, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (256, ), (1, ))
    assert_size_stride(arg525_1, (256, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (256, ), (1, ))
    assert_size_stride(arg528_1, (256, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (512, ), (1, ))
    assert_size_stride(arg531_1, (512, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (256, ), (1, ))
    assert_size_stride(arg534_1, (256, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (256, ), (1, ))
    assert_size_stride(arg537_1, (256, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (512, ), (1, ))
    assert_size_stride(arg540_1, (512, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (512, ), (1, ))
    assert_size_stride(arg543_1, (512, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (256, ), (1, ))
    assert_size_stride(arg546_1, (256, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (256, ), (1, ))
    assert_size_stride(arg549_1, (256, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (512, ), (1, ))
    assert_size_stride(arg552_1, (512, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (256, ), (1, ))
    assert_size_stride(arg555_1, (256, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (256, ), (1, ))
    assert_size_stride(arg558_1, (256, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (512, ), (1, ))
    assert_size_stride(arg561_1, (512, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (512, ), (1, ))
    assert_size_stride(arg564_1, (512, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (256, ), (1, ))
    assert_size_stride(arg567_1, (256, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (256, ), (1, ))
    assert_size_stride(arg570_1, (256, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (512, ), (1, ))
    assert_size_stride(arg573_1, (512, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (256, ), (1, ))
    assert_size_stride(arg576_1, (256, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (256, ), (1, ))
    assert_size_stride(arg579_1, (256, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (512, ), (1, ))
    assert_size_stride(arg582_1, (512, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (512, ), (1, ))
    assert_size_stride(arg585_1, (512, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (256, ), (1, ))
    assert_size_stride(arg588_1, (256, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (256, ), (1, ))
    assert_size_stride(arg591_1, (256, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (512, ), (1, ))
    assert_size_stride(arg594_1, (512, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (256, ), (1, ))
    assert_size_stride(arg597_1, (256, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (256, ), (1, ))
    assert_size_stride(arg600_1, (256, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (512, ), (1, ))
    assert_size_stride(arg603_1, (512, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (512, ), (1, ))
    assert_size_stride(arg606_1, (512, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (1024, ), (1, ))
    assert_size_stride(arg609_1, (1024, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (512, ), (1, ))
    assert_size_stride(arg612_1, (512, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (512, ), (1, ))
    assert_size_stride(arg615_1, (512, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (1024, ), (1, ))
    assert_size_stride(arg618_1, (1024, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (512, ), (1, ))
    assert_size_stride(arg621_1, (512, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (512, ), (1, ))
    assert_size_stride(arg624_1, (512, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (1024, ), (1, ))
    assert_size_stride(arg628_1, (), ())
    assert_size_stride(arg629_1, (1024, ), (1, ))
    assert_size_stride(arg630_1, (1024, ), (1, ))
    assert_size_stride(arg631_1, (), ())
    assert_size_stride(arg632_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg632_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg632_1
        buf1 = empty_strided((16, 3, 7, 7), (147, 1, 21, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 49, grid=grid(48, 49), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___base_layer_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 224, 224), (802816, 50176, 224, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 16, 224, 224), (802816, 1, 3584, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg317_1, arg318_1, arg1_1, arg2_1, buf3, 128, 50176, grid=grid(128, 50176), stream=stream0)
        del arg1_1
        del arg2_1
        del arg317_1
        del arg318_1
        del buf2
        buf4 = empty_strided((16, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___base_layer_1, l__mod___level0_0, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg3_1, buf4, 256, 9, grid=grid(256, 9), stream=stream0)
        del arg3_1
        # Source Nodes: [l__mod___base_layer_1, l__mod___level0_0, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 16, 224, 224), (802816, 50176, 224, 1))
        del buf4
        buf6 = buf3; del buf3  # reuse
        # Source Nodes: [l__mod___level0_1, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, arg320_1, arg321_1, arg4_1, arg5_1, buf6, 128, 50176, grid=grid(128, 50176), stream=stream0)
        del arg320_1
        del arg321_1
        del arg4_1
        del arg5_1
        del buf5
        buf7 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level0_1, l__mod___level1_0, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4.run(arg6_1, buf7, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [l__mod___level0_1, l__mod___level1_0, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf7
        buf9 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___level1_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, arg323_1, arg324_1, arg7_1, arg8_1, buf9, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg323_1
        del arg324_1
        del arg7_1
        del arg8_1
        del buf8
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg12_1
        buf11 = reinterpret_tensor(buf6, (8, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf6  # reuse
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf10, arg329_1, arg330_1, arg13_1, arg14_1, buf11, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del arg13_1
        del arg14_1
        del arg329_1
        del arg330_1
        del buf10
        buf12 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(arg15_1, buf12, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg15_1
        # Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf14 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf13, arg332_1, arg333_1, arg16_1, arg17_1, buf14, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        del arg332_1
        del arg333_1
        del buf13
        # Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf15 = extern_kernels.convolution(buf14, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg18_1
        buf16 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [bottom], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_9.run(buf9, buf16, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [bottom, l__mod___level2_project_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf17 = extern_kernels.convolution(buf16, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg9_1
        buf18 = buf15; del buf15  # reuse
        buf19 = reinterpret_tensor(buf9, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf9  # reuse
        # Source Nodes: [out_7, out_8, shortcut, shortcut_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf18, arg335_1, arg336_1, arg19_1, arg20_1, buf17, arg326_1, arg327_1, arg10_1, arg11_1, buf19, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg19_1
        del arg20_1
        del arg326_1
        del arg327_1
        del arg335_1
        del arg336_1
        del buf17
        del buf18
        # Source Nodes: [out_10, shortcut_1], Original ATen: [aten.convolution, aten.relu]
        buf20 = extern_kernels.convolution(buf19, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg21_1
        buf21 = buf14; del buf14  # reuse
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf20, arg338_1, arg339_1, arg22_1, arg23_1, buf21, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del arg22_1
        del arg23_1
        del arg338_1
        del arg339_1
        del buf20
        buf22 = buf12; del buf12  # reuse
        # Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(arg24_1, buf22, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg24_1
        # Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf23 = extern_kernels.convolution(buf21, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del buf22
        buf24 = buf21; del buf21  # reuse
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf23, arg341_1, arg342_1, arg25_1, arg26_1, buf24, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del arg25_1
        del arg26_1
        del arg341_1
        del arg342_1
        del buf23
        # Source Nodes: [out_14, out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf24, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg27_1
        buf28 = reinterpret_tensor(buf11, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf11  # reuse
        buf26 = reinterpret_tensor(buf28, (8, 128, 56, 56), (802816, 1, 14336, 256), 0)  # alias
        buf27 = reinterpret_tensor(buf28, (8, 128, 56, 56), (802816, 1, 14336, 256), 128)  # alias
        # Source Nodes: [cat_27, out_17, out_18, x2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_11.run(buf25, arg344_1, arg345_1, arg28_1, arg29_1, buf19, buf26, buf27, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del arg28_1
        del arg29_1
        del arg344_1
        del arg345_1
        del buf19
        del buf27
        # Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg30_1
        buf30 = reinterpret_tensor(buf25, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf25  # reuse
        # Source Nodes: [x_4, x_5, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf29, arg347_1, arg348_1, arg31_1, arg32_1, buf26, buf30, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del arg31_1
        del arg32_1
        del arg347_1
        del arg348_1
        del buf26
        del buf28
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg36_1
        buf32 = reinterpret_tensor(buf29, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf29  # reuse
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf31, arg353_1, arg354_1, arg37_1, arg38_1, buf32, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del arg353_1
        del arg354_1
        del arg37_1
        del arg38_1
        del buf31
        buf33 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg39_1, buf33, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg39_1
        # Source Nodes: [out_21, out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf35 = reinterpret_tensor(buf16, (8, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf16  # reuse
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf34, arg356_1, arg357_1, arg40_1, arg41_1, buf35, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg356_1
        del arg357_1
        del arg40_1
        del arg41_1
        del buf34
        # Source Nodes: [out_24, out_25, out_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg42_1
        buf37 = buf35; del buf35  # reuse
        # Source Nodes: [bottom_3], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_16.run(buf30, buf37, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [bottom_3, l__mod___level3_tree1_tree1_project_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf38 = extern_kernels.convolution(buf37, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg33_1
        buf39 = buf36; del buf36  # reuse
        buf40 = reinterpret_tensor(buf24, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf24  # reuse
        # Source Nodes: [out_27, out_28, shortcut_4, shortcut_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf39, arg359_1, arg360_1, arg43_1, arg44_1, buf38, arg350_1, arg351_1, arg34_1, arg35_1, buf40, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del arg34_1
        del arg350_1
        del arg351_1
        del arg359_1
        del arg35_1
        del arg360_1
        del arg43_1
        del arg44_1
        del buf38
        del buf39
        # Source Nodes: [out_30, shortcut_5], Original ATen: [aten.convolution, aten.relu]
        buf41 = extern_kernels.convolution(buf40, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg45_1
        buf42 = buf37; del buf37  # reuse
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf41, arg362_1, arg363_1, arg46_1, arg47_1, buf42, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg362_1
        del arg363_1
        del arg46_1
        del arg47_1
        del buf41
        buf43 = buf33; del buf33  # reuse
        # Source Nodes: [out_31, out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg48_1, buf43, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg48_1
        # Source Nodes: [out_31, out_32, out_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf44 = extern_kernels.convolution(buf42, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf45 = buf42; del buf42  # reuse
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf44, arg365_1, arg366_1, arg49_1, arg50_1, buf45, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg365_1
        del arg366_1
        del arg49_1
        del arg50_1
        del buf44
        # Source Nodes: [out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg51_1
        buf49 = reinterpret_tensor(buf32, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf32  # reuse
        buf47 = reinterpret_tensor(buf49, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf48 = reinterpret_tensor(buf49, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        # Source Nodes: [cat_26, out_37, out_38, x2_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18.run(buf46, arg368_1, arg369_1, arg52_1, arg53_1, buf40, buf47, buf48, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg368_1
        del arg369_1
        del arg52_1
        del arg53_1
        del buf40
        del buf48
        # Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg54_1
        buf51 = reinterpret_tensor(buf46, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf46  # reuse
        # Source Nodes: [x1_2, x_10, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf50, arg371_1, arg372_1, arg55_1, arg56_1, buf47, buf51, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg371_1
        del arg372_1
        del arg55_1
        del arg56_1
        del buf47
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg57_1
        buf53 = buf45; del buf45  # reuse
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf52, arg374_1, arg375_1, arg58_1, arg59_1, buf53, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg374_1
        del arg375_1
        del arg58_1
        del arg59_1
        del buf52
        buf54 = buf43; del buf43  # reuse
        # Source Nodes: [out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg60_1, buf54, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg60_1
        # Source Nodes: [out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf56 = buf53; del buf53  # reuse
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf55, arg377_1, arg378_1, arg61_1, arg62_1, buf56, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg377_1
        del arg378_1
        del arg61_1
        del arg62_1
        del buf55
        # Source Nodes: [out_44, out_45, out_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf57 = extern_kernels.convolution(buf56, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg63_1
        buf58 = reinterpret_tensor(buf50, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf50  # reuse
        buf68 = empty_strided((8, 768, 28, 28), (602112, 1, 21504, 768), device='cuda', dtype=torch.float32)
        buf67 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 512)  # alias
        # Source Nodes: [cat_25, out_47, out_48, shortcut_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_20.run(buf57, arg380_1, arg381_1, arg64_1, arg65_1, buf51, buf58, buf67, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg380_1
        del arg381_1
        del arg64_1
        del arg65_1
        del buf51
        del buf57
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg66_1
        buf60 = buf56; del buf56  # reuse
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf59, arg383_1, arg384_1, arg67_1, arg68_1, buf60, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg383_1
        del arg384_1
        del arg67_1
        del arg68_1
        del buf59
        buf61 = buf54; del buf54  # reuse
        # Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg69_1, buf61, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg69_1
        # Source Nodes: [out_51, out_52, out_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf63 = buf60; del buf60  # reuse
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf62, arg386_1, arg387_1, arg70_1, arg71_1, buf63, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg386_1
        del arg387_1
        del arg70_1
        del arg71_1
        del buf62
        # Source Nodes: [out_54, out_55, out_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg72_1
        buf65 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 0)  # alias
        buf66 = reinterpret_tensor(buf68, (8, 256, 28, 28), (602112, 1, 21504, 768), 256)  # alias
        # Source Nodes: [cat_25, out_57, out_58, x2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_21.run(buf64, arg389_1, arg390_1, arg73_1, arg74_1, buf58, buf65, buf66, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg389_1
        del arg390_1
        del arg73_1
        del arg74_1
        del buf58
        del buf66
        del buf67
        # Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg75_1
        buf70 = reinterpret_tensor(buf64, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf64  # reuse
        # Source Nodes: [x1_4, x_15, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf69, arg392_1, arg393_1, arg76_1, arg77_1, buf65, buf70, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg392_1
        del arg393_1
        del arg76_1
        del arg77_1
        del buf65
        del buf68
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg78_1
        buf72 = buf63; del buf63  # reuse
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf71, arg395_1, arg396_1, arg79_1, arg80_1, buf72, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg395_1
        del arg396_1
        del arg79_1
        del arg80_1
        del buf71
        buf73 = buf61; del buf61  # reuse
        # Source Nodes: [out_61, out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg81_1, buf73, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg81_1
        # Source Nodes: [out_61, out_62, out_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf72, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf75 = buf72; del buf72  # reuse
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf74, arg398_1, arg399_1, arg82_1, arg83_1, buf75, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg398_1
        del arg399_1
        del arg82_1
        del arg83_1
        del buf74
        # Source Nodes: [out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg84_1
        buf77 = reinterpret_tensor(buf69, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf69  # reuse
        buf107 = empty_strided((8, 1152, 28, 28), (903168, 1, 32256, 1152), device='cuda', dtype=torch.float32)
        buf105 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 640)  # alias
        # Source Nodes: [cat_23, out_67, out_68, shortcut_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23.run(buf76, arg401_1, arg402_1, arg85_1, arg86_1, buf70, buf77, buf105, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg401_1
        del arg402_1
        del arg85_1
        del arg86_1
        del buf70
        del buf76
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg87_1
        buf79 = buf75; del buf75  # reuse
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf78, arg404_1, arg405_1, arg88_1, arg89_1, buf79, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg404_1
        del arg405_1
        del arg88_1
        del arg89_1
        del buf78
        buf80 = buf73; del buf73  # reuse
        # Source Nodes: [out_71, out_72, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg90_1, buf80, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg90_1
        # Source Nodes: [out_71, out_72, out_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf82 = buf79; del buf79  # reuse
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf81, arg407_1, arg408_1, arg91_1, arg92_1, buf82, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg407_1
        del arg408_1
        del arg91_1
        del arg92_1
        del buf81
        # Source Nodes: [out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf83 = extern_kernels.convolution(buf82, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg93_1
        buf86 = buf49; del buf49  # reuse
        buf84 = reinterpret_tensor(buf86, (8, 256, 28, 28), (401408, 1, 14336, 512), 0)  # alias
        buf85 = reinterpret_tensor(buf86, (8, 256, 28, 28), (401408, 1, 14336, 512), 256)  # alias
        # Source Nodes: [cat_24, out_77, out_78, x2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_18.run(buf83, arg410_1, arg411_1, arg94_1, arg95_1, buf77, buf84, buf85, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg410_1
        del arg411_1
        del arg94_1
        del arg95_1
        del buf77
        del buf85
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg96_1
        buf88 = reinterpret_tensor(buf83, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf83  # reuse
        # Source Nodes: [x1_6, x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf87, arg413_1, arg414_1, arg97_1, arg98_1, buf84, buf88, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg413_1
        del arg414_1
        del arg97_1
        del arg98_1
        del buf84
        del buf86
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg99_1
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf89, arg416_1, arg417_1, arg100_1, arg101_1, buf90, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg100_1
        del arg101_1
        del arg416_1
        del arg417_1
        del buf89
        buf91 = buf80; del buf80  # reuse
        # Source Nodes: [out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg102_1, buf91, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg102_1
        # Source Nodes: [out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf93 = buf90; del buf90  # reuse
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf92, arg419_1, arg420_1, arg103_1, arg104_1, buf93, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg103_1
        del arg104_1
        del arg419_1
        del arg420_1
        del buf92
        # Source Nodes: [out_84, out_85, out_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf94 = extern_kernels.convolution(buf93, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg105_1
        buf95 = reinterpret_tensor(buf87, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf87  # reuse
        buf106 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 896)  # alias
        # Source Nodes: [cat_23, out_87, out_88, shortcut_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_23.run(buf94, arg422_1, arg423_1, arg106_1, arg107_1, buf88, buf95, buf106, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg106_1
        del arg107_1
        del arg422_1
        del arg423_1
        del buf88
        del buf94
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 128, 28, 28), (100352, 784, 28, 1))
        del arg108_1
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf96, arg425_1, arg426_1, arg109_1, arg110_1, buf97, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg109_1
        del arg110_1
        del arg425_1
        del arg426_1
        del buf96
        buf98 = buf91; del buf91  # reuse
        # Source Nodes: [out_91, out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg111_1, buf98, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg111_1
        # Source Nodes: [out_91, out_92, out_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf99 = extern_kernels.convolution(buf97, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 128, 28, 28), (100352, 784, 28, 1))
        del buf98
        buf100 = buf97; del buf97  # reuse
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf99, arg428_1, arg429_1, arg112_1, arg113_1, buf100, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del arg112_1
        del arg113_1
        del arg428_1
        del arg429_1
        del buf99
        # Source Nodes: [out_94, out_95, out_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf101 = extern_kernels.convolution(buf100, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg114_1
        buf102 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 0)  # alias
        buf103 = reinterpret_tensor(buf107, (8, 256, 28, 28), (903168, 1, 32256, 1152), 256)  # alias
        # Source Nodes: [cat_23, out_97, out_98, x2_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_24.run(buf101, arg431_1, arg432_1, arg115_1, arg116_1, buf95, buf102, buf103, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg115_1
        del arg116_1
        del arg431_1
        del arg432_1
        del buf101
        buf104 = reinterpret_tensor(buf107, (8, 128, 28, 28), (903168, 1, 32256, 1152), 512)  # alias
        # Source Nodes: [bottom_1], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_25.run(buf30, buf104, 802816, grid=grid(802816), stream=stream0)
        del buf103
        del buf104
        del buf105
        del buf106
        # Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg117_1
        buf109 = buf95; del buf95  # reuse
        # Source Nodes: [x_26, x_27, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_26.run(buf108, arg434_1, arg435_1, arg118_1, arg119_1, buf102, buf109, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del arg118_1
        del arg119_1
        del arg434_1
        del arg435_1
        del buf102
        del buf107
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 256, 28, 28), (200704, 784, 28, 1))
        del arg123_1
        buf111 = reinterpret_tensor(buf108, (8, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf108  # reuse
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf110, arg440_1, arg441_1, arg124_1, arg125_1, buf111, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del arg124_1
        del arg125_1
        del arg440_1
        del arg441_1
        del buf110
        buf112 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101, out_102, out_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg126_1, buf112, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg126_1
        # Source Nodes: [out_101, out_102, out_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf114 = empty_strided((8, 256, 14, 14), (50176, 1, 3584, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf113, arg443_1, arg444_1, arg127_1, arg128_1, buf114, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg127_1
        del arg128_1
        del arg443_1
        del arg444_1
        del buf113
        # Source Nodes: [out_104, out_105, out_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf115 = extern_kernels.convolution(buf114, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg129_1
        buf116 = buf114; del buf114  # reuse
        # Source Nodes: [bottom_11], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_30.run(buf109, buf116, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [bottom_11, l__mod___level4_tree1_tree1_tree1_project_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf117 = extern_kernels.convolution(buf116, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg120_1
        buf118 = buf115; del buf115  # reuse
        buf119 = reinterpret_tensor(buf100, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf100  # reuse
        # Source Nodes: [out_107, out_108, shortcut_16, shortcut_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf118, arg446_1, arg447_1, arg130_1, arg131_1, buf117, arg437_1, arg438_1, arg121_1, arg122_1, buf119, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del arg121_1
        del arg122_1
        del arg130_1
        del arg131_1
        del arg437_1
        del arg438_1
        del arg446_1
        del arg447_1
        del buf117
        del buf118
        # Source Nodes: [out_110, shortcut_17], Original ATen: [aten.convolution, aten.relu]
        buf120 = extern_kernels.convolution(buf119, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg132_1
        buf121 = buf116; del buf116  # reuse
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf120, arg449_1, arg450_1, arg133_1, arg134_1, buf121, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg133_1
        del arg134_1
        del arg449_1
        del arg450_1
        del buf120
        buf122 = buf112; del buf112  # reuse
        # Source Nodes: [out_111, out_112, out_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg135_1, buf122, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg135_1
        # Source Nodes: [out_111, out_112, out_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf124 = buf121; del buf121  # reuse
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf123, arg452_1, arg453_1, arg136_1, arg137_1, buf124, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg136_1
        del arg137_1
        del arg452_1
        del arg453_1
        del buf123
        # Source Nodes: [out_114, out_115, out_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf125 = extern_kernels.convolution(buf124, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg138_1
        buf128 = reinterpret_tensor(buf111, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf111  # reuse
        buf126 = reinterpret_tensor(buf128, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf127 = reinterpret_tensor(buf128, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Source Nodes: [cat_22, out_117, out_118, x2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf125, arg455_1, arg456_1, arg139_1, arg140_1, buf119, buf126, buf127, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg139_1
        del arg140_1
        del arg455_1
        del arg456_1
        del buf119
        del buf127
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg141_1
        buf130 = reinterpret_tensor(buf125, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf125  # reuse
        # Source Nodes: [x1_9, x_34, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf129, arg458_1, arg459_1, arg142_1, arg143_1, buf126, buf130, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg142_1
        del arg143_1
        del arg458_1
        del arg459_1
        del buf126
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg144_1
        buf132 = buf124; del buf124  # reuse
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf131, arg461_1, arg462_1, arg145_1, arg146_1, buf132, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg145_1
        del arg146_1
        del arg461_1
        del arg462_1
        del buf131
        buf133 = buf122; del buf122  # reuse
        # Source Nodes: [out_121, out_122, out_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg147_1, buf133, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg147_1
        # Source Nodes: [out_121, out_122, out_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf135 = buf132; del buf132  # reuse
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf134, arg464_1, arg465_1, arg148_1, arg149_1, buf135, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg148_1
        del arg149_1
        del arg464_1
        del arg465_1
        del buf134
        # Source Nodes: [out_124, out_125, out_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf136 = extern_kernels.convolution(buf135, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg150_1
        buf137 = reinterpret_tensor(buf129, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf129  # reuse
        buf147 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cuda', dtype=torch.float32)
        buf146 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Source Nodes: [cat_21, out_127, out_128, shortcut_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34.run(buf136, arg467_1, arg468_1, arg151_1, arg152_1, buf130, buf137, buf146, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg151_1
        del arg152_1
        del arg467_1
        del arg468_1
        del buf130
        del buf136
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg153_1
        buf139 = buf135; del buf135  # reuse
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf138, arg470_1, arg471_1, arg154_1, arg155_1, buf139, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg154_1
        del arg155_1
        del arg470_1
        del arg471_1
        del buf138
        buf140 = buf133; del buf133  # reuse
        # Source Nodes: [out_131, out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg156_1, buf140, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg156_1
        # Source Nodes: [out_131, out_132, out_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf141 = extern_kernels.convolution(buf139, buf140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf142 = buf139; del buf139  # reuse
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf141, arg473_1, arg474_1, arg157_1, arg158_1, buf142, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg157_1
        del arg158_1
        del arg473_1
        del arg474_1
        del buf141
        # Source Nodes: [out_134, out_135, out_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf143 = extern_kernels.convolution(buf142, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg159_1
        buf144 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf145 = reinterpret_tensor(buf147, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        # Source Nodes: [cat_21, out_137, out_138, x2_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35.run(buf143, arg476_1, arg477_1, arg160_1, arg161_1, buf137, buf144, buf145, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg160_1
        del arg161_1
        del arg476_1
        del arg477_1
        del buf137
        del buf145
        del buf146
        # Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg162_1
        buf149 = reinterpret_tensor(buf143, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf143  # reuse
        # Source Nodes: [x1_11, x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf148, arg479_1, arg480_1, arg163_1, arg164_1, buf144, buf149, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg163_1
        del arg164_1
        del arg479_1
        del arg480_1
        del buf144
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg165_1
        buf151 = buf142; del buf142  # reuse
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf150, arg482_1, arg483_1, arg166_1, arg167_1, buf151, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg166_1
        del arg167_1
        del arg482_1
        del arg483_1
        del buf150
        buf152 = buf140; del buf140  # reuse
        # Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg168_1, buf152, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg168_1
        # Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf153 = extern_kernels.convolution(buf151, buf152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf154 = buf151; del buf151  # reuse
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf153, arg485_1, arg486_1, arg169_1, arg170_1, buf154, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg169_1
        del arg170_1
        del arg485_1
        del arg486_1
        del buf153
        # Source Nodes: [out_144, out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf155 = extern_kernels.convolution(buf154, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg171_1
        buf156 = reinterpret_tensor(buf148, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf148  # reuse
        buf185 = reinterpret_tensor(buf30, (8, 2048, 14, 14), (401408, 1, 28672, 2048), 0); del buf30  # reuse
        buf183 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1024)  # alias
        # Source Nodes: [cat_19, out_147, out_148, shortcut_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37.run(buf155, arg488_1, arg489_1, arg172_1, arg173_1, buf149, buf156, buf183, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg172_1
        del arg173_1
        del arg488_1
        del arg489_1
        del buf149
        del buf155
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg174_1
        buf158 = buf154; del buf154  # reuse
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf157, arg491_1, arg492_1, arg175_1, arg176_1, buf158, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg175_1
        del arg176_1
        del arg491_1
        del arg492_1
        del buf157
        buf159 = buf152; del buf152  # reuse
        # Source Nodes: [out_151, out_152, out_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg177_1, buf159, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg177_1
        # Source Nodes: [out_151, out_152, out_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf161 = buf158; del buf158  # reuse
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf160, arg494_1, arg495_1, arg178_1, arg179_1, buf161, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg178_1
        del arg179_1
        del arg494_1
        del arg495_1
        del buf160
        # Source Nodes: [out_154, out_155, out_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf162 = extern_kernels.convolution(buf161, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg180_1
        buf165 = buf128; del buf128  # reuse
        buf163 = reinterpret_tensor(buf165, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf164 = reinterpret_tensor(buf165, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Source Nodes: [cat_20, out_157, out_158, x2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf162, arg497_1, arg498_1, arg181_1, arg182_1, buf156, buf163, buf164, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg181_1
        del arg182_1
        del arg497_1
        del arg498_1
        del buf156
        del buf164
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg183_1
        buf167 = reinterpret_tensor(buf162, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf162  # reuse
        # Source Nodes: [x1_13, x_45, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf166, arg500_1, arg501_1, arg184_1, arg185_1, buf163, buf167, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg184_1
        del arg185_1
        del arg500_1
        del arg501_1
        del buf163
        # Source Nodes: [out_160], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg186_1
        buf169 = buf161; del buf161  # reuse
        # Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf168, arg503_1, arg504_1, arg187_1, arg188_1, buf169, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg187_1
        del arg188_1
        del arg503_1
        del arg504_1
        del buf168
        buf170 = buf159; del buf159  # reuse
        # Source Nodes: [out_161, out_162, out_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg189_1, buf170, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg189_1
        # Source Nodes: [out_161, out_162, out_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf171 = extern_kernels.convolution(buf169, buf170, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf172 = buf169; del buf169  # reuse
        # Source Nodes: [out_164, out_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf171, arg506_1, arg507_1, arg190_1, arg191_1, buf172, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg190_1
        del arg191_1
        del arg506_1
        del arg507_1
        del buf171
        # Source Nodes: [out_164, out_165, out_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf173 = extern_kernels.convolution(buf172, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg192_1
        buf174 = reinterpret_tensor(buf166, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf166  # reuse
        buf184 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 1536)  # alias
        # Source Nodes: [cat_19, out_167, out_168, shortcut_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_37.run(buf173, arg509_1, arg510_1, arg193_1, arg194_1, buf167, buf174, buf184, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg193_1
        del arg194_1
        del arg509_1
        del arg510_1
        del buf167
        del buf173
        # Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg195_1
        buf176 = buf172; del buf172  # reuse
        # Source Nodes: [out_171, out_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf175, arg512_1, arg513_1, arg196_1, arg197_1, buf176, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg196_1
        del arg197_1
        del arg512_1
        del arg513_1
        del buf175
        buf177 = buf170; del buf170  # reuse
        # Source Nodes: [out_171, out_172, out_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg198_1, buf177, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg198_1
        # Source Nodes: [out_171, out_172, out_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf179 = buf176; del buf176  # reuse
        # Source Nodes: [out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf178, arg515_1, arg516_1, arg199_1, arg200_1, buf179, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg199_1
        del arg200_1
        del arg515_1
        del arg516_1
        del buf178
        # Source Nodes: [out_174, out_175, out_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf180 = extern_kernels.convolution(buf179, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg201_1
        buf181 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 0)  # alias
        buf182 = reinterpret_tensor(buf185, (8, 512, 14, 14), (401408, 1, 28672, 2048), 512)  # alias
        # Source Nodes: [cat_19, out_177, out_178, x2_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_38.run(buf180, arg518_1, arg519_1, arg202_1, arg203_1, buf174, buf181, buf182, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg202_1
        del arg203_1
        del arg518_1
        del arg519_1
        del buf174
        del buf182
        del buf183
        del buf184
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg204_1
        buf187 = reinterpret_tensor(buf180, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf180  # reuse
        buf262 = empty_strided((8, 2816, 14, 14), (551936, 1, 39424, 2816), device='cuda', dtype=torch.float32)
        buf259 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1280)  # alias
        # Source Nodes: [cat_15, x1_15, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_39.run(buf186, arg521_1, arg522_1, arg205_1, arg206_1, buf181, buf187, buf259, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg205_1
        del arg206_1
        del arg521_1
        del arg522_1
        del buf181
        del buf185
        del buf186
        # Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg207_1
        buf189 = buf179; del buf179  # reuse
        # Source Nodes: [out_181, out_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf188, arg524_1, arg525_1, arg208_1, arg209_1, buf189, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg208_1
        del arg209_1
        del arg524_1
        del arg525_1
        del buf188
        buf190 = buf177; del buf177  # reuse
        # Source Nodes: [out_181, out_182, out_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg210_1, buf190, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg210_1
        # Source Nodes: [out_181, out_182, out_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf192 = buf189; del buf189  # reuse
        # Source Nodes: [out_184, out_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf191, arg527_1, arg528_1, arg211_1, arg212_1, buf192, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg211_1
        del arg212_1
        del arg527_1
        del arg528_1
        del buf191
        # Source Nodes: [out_184, out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf193 = extern_kernels.convolution(buf192, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg213_1
        buf194 = buf187; del buf187  # reuse
        # Source Nodes: [out_187, out_188, shortcut_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_40.run(buf194, buf193, arg530_1, arg531_1, arg214_1, arg215_1, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg214_1
        del arg215_1
        del arg530_1
        del arg531_1
        del buf193
        # Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg216_1
        buf196 = buf192; del buf192  # reuse
        # Source Nodes: [out_191, out_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf195, arg533_1, arg534_1, arg217_1, arg218_1, buf196, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg217_1
        del arg218_1
        del arg533_1
        del arg534_1
        del buf195
        buf197 = buf190; del buf190  # reuse
        # Source Nodes: [out_191, out_192, out_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg219_1, buf197, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg219_1
        # Source Nodes: [out_191, out_192, out_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf199 = buf196; del buf196  # reuse
        # Source Nodes: [out_194, out_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf198, arg536_1, arg537_1, arg220_1, arg221_1, buf199, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg220_1
        del arg221_1
        del arg536_1
        del arg537_1
        del buf198
        # Source Nodes: [out_194, out_195, out_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf200 = extern_kernels.convolution(buf199, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg222_1
        buf203 = buf165; del buf165  # reuse
        buf201 = reinterpret_tensor(buf203, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf202 = reinterpret_tensor(buf203, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Source Nodes: [cat_18, out_197, out_198, x2_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf200, arg539_1, arg540_1, arg223_1, arg224_1, buf194, buf201, buf202, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg223_1
        del arg224_1
        del arg539_1
        del arg540_1
        del buf194
        del buf202
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg225_1
        buf205 = reinterpret_tensor(buf200, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf200  # reuse
        # Source Nodes: [x1_17, x_57, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf204, arg542_1, arg543_1, arg226_1, arg227_1, buf201, buf205, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg226_1
        del arg227_1
        del arg542_1
        del arg543_1
        del buf201
        # Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg228_1
        buf207 = buf199; del buf199  # reuse
        # Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf206, arg545_1, arg546_1, arg229_1, arg230_1, buf207, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg229_1
        del arg230_1
        del arg545_1
        del arg546_1
        del buf206
        buf208 = buf197; del buf197  # reuse
        # Source Nodes: [out_201, out_202, out_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg231_1, buf208, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg231_1
        # Source Nodes: [out_201, out_202, out_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf210 = buf207; del buf207  # reuse
        # Source Nodes: [out_204, out_205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf209, arg548_1, arg549_1, arg232_1, arg233_1, buf210, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg232_1
        del arg233_1
        del arg548_1
        del arg549_1
        del buf209
        # Source Nodes: [out_204, out_205, out_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf211 = extern_kernels.convolution(buf210, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg234_1
        buf212 = reinterpret_tensor(buf204, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf204  # reuse
        buf222 = buf147; del buf147  # reuse
        buf221 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 1024)  # alias
        # Source Nodes: [cat_17, out_207, out_208, shortcut_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_34.run(buf211, arg551_1, arg552_1, arg235_1, arg236_1, buf205, buf212, buf221, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg235_1
        del arg236_1
        del arg551_1
        del arg552_1
        del buf205
        del buf211
        # Source Nodes: [out_210], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg237_1
        buf214 = buf210; del buf210  # reuse
        # Source Nodes: [out_211, out_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf213, arg554_1, arg555_1, arg238_1, arg239_1, buf214, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg238_1
        del arg239_1
        del arg554_1
        del arg555_1
        del buf213
        buf215 = buf208; del buf208  # reuse
        # Source Nodes: [out_211, out_212, out_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg240_1, buf215, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg240_1
        # Source Nodes: [out_211, out_212, out_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf216 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf217 = buf214; del buf214  # reuse
        # Source Nodes: [out_214, out_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf216, arg557_1, arg558_1, arg241_1, arg242_1, buf217, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg241_1
        del arg242_1
        del arg557_1
        del arg558_1
        del buf216
        # Source Nodes: [out_214, out_215, out_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf218 = extern_kernels.convolution(buf217, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg243_1
        buf219 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 0)  # alias
        buf220 = reinterpret_tensor(buf222, (8, 512, 14, 14), (301056, 1, 21504, 1536), 512)  # alias
        # Source Nodes: [cat_17, out_217, out_218, x2_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_35.run(buf218, arg560_1, arg561_1, arg244_1, arg245_1, buf212, buf219, buf220, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg244_1
        del arg245_1
        del arg560_1
        del arg561_1
        del buf212
        del buf220
        del buf221
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg246_1
        buf224 = reinterpret_tensor(buf218, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf218  # reuse
        # Source Nodes: [x1_19, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf223, arg563_1, arg564_1, arg247_1, arg248_1, buf219, buf224, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg247_1
        del arg248_1
        del arg563_1
        del arg564_1
        del buf219
        del buf222
        # Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg249_1
        buf226 = buf217; del buf217  # reuse
        # Source Nodes: [out_221, out_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf225, arg566_1, arg567_1, arg250_1, arg251_1, buf226, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg250_1
        del arg251_1
        del arg566_1
        del arg567_1
        del buf225
        buf227 = buf215; del buf215  # reuse
        # Source Nodes: [out_221, out_222, out_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg252_1, buf227, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg252_1
        # Source Nodes: [out_221, out_222, out_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf229 = buf226; del buf226  # reuse
        # Source Nodes: [out_224, out_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf228, arg569_1, arg570_1, arg253_1, arg254_1, buf229, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg253_1
        del arg254_1
        del arg569_1
        del arg570_1
        del buf228
        # Source Nodes: [out_224, out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf230 = extern_kernels.convolution(buf229, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg255_1
        buf231 = reinterpret_tensor(buf223, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf223  # reuse
        buf260 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 1792)  # alias
        # Source Nodes: [cat_15, out_227, out_228, shortcut_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41.run(buf230, arg572_1, arg573_1, arg256_1, arg257_1, buf224, buf231, buf260, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg256_1
        del arg257_1
        del arg572_1
        del arg573_1
        del buf224
        del buf230
        # Source Nodes: [out_230], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg258_1
        buf233 = buf229; del buf229  # reuse
        # Source Nodes: [out_231, out_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf232, arg575_1, arg576_1, arg259_1, arg260_1, buf233, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg259_1
        del arg260_1
        del arg575_1
        del arg576_1
        del buf232
        buf234 = buf227; del buf227  # reuse
        # Source Nodes: [out_231, out_232, out_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg261_1, buf234, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg261_1
        # Source Nodes: [out_231, out_232, out_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf236 = buf233; del buf233  # reuse
        # Source Nodes: [out_234, out_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf235, arg578_1, arg579_1, arg262_1, arg263_1, buf236, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg262_1
        del arg263_1
        del arg578_1
        del arg579_1
        del buf235
        # Source Nodes: [out_234, out_235, out_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf237 = extern_kernels.convolution(buf236, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg264_1
        buf240 = buf203; del buf203  # reuse
        buf238 = reinterpret_tensor(buf240, (8, 512, 14, 14), (200704, 1, 14336, 1024), 0)  # alias
        buf239 = reinterpret_tensor(buf240, (8, 512, 14, 14), (200704, 1, 14336, 1024), 512)  # alias
        # Source Nodes: [cat_16, out_237, out_238, x2_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_32.run(buf237, arg581_1, arg582_1, arg265_1, arg266_1, buf231, buf238, buf239, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg265_1
        del arg266_1
        del arg581_1
        del arg582_1
        del buf231
        del buf239
        # Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg267_1
        buf242 = reinterpret_tensor(buf237, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf237  # reuse
        # Source Nodes: [x1_21, x_68, x_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf241, arg584_1, arg585_1, arg268_1, arg269_1, buf238, buf242, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg268_1
        del arg269_1
        del arg584_1
        del arg585_1
        del buf238
        del buf240
        # Source Nodes: [out_240], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg270_1
        buf244 = buf236; del buf236  # reuse
        # Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf243, arg587_1, arg588_1, arg271_1, arg272_1, buf244, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg271_1
        del arg272_1
        del arg587_1
        del arg588_1
        del buf243
        buf245 = buf234; del buf234  # reuse
        # Source Nodes: [out_241, out_242, out_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg273_1, buf245, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg273_1
        # Source Nodes: [out_241, out_242, out_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf246 = extern_kernels.convolution(buf244, buf245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 256, 14, 14), (50176, 196, 14, 1))
        buf247 = buf244; del buf244  # reuse
        # Source Nodes: [out_244, out_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf246, arg590_1, arg591_1, arg274_1, arg275_1, buf247, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg274_1
        del arg275_1
        del arg590_1
        del arg591_1
        del buf246
        # Source Nodes: [out_244, out_245, out_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf248 = extern_kernels.convolution(buf247, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg276_1
        buf249 = reinterpret_tensor(buf241, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf241  # reuse
        buf261 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 2304)  # alias
        # Source Nodes: [cat_15, out_247, out_248, shortcut_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_41.run(buf248, arg593_1, arg594_1, arg277_1, arg278_1, buf242, buf249, buf261, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg277_1
        del arg278_1
        del arg593_1
        del arg594_1
        del buf242
        del buf248
        # Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg279_1
        buf251 = buf247; del buf247  # reuse
        # Source Nodes: [out_251, out_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf250, arg596_1, arg597_1, arg280_1, arg281_1, buf251, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg280_1
        del arg281_1
        del arg596_1
        del arg597_1
        del buf250
        buf252 = buf245; del buf245  # reuse
        # Source Nodes: [out_251, out_252, out_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(arg282_1, buf252, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg282_1
        # Source Nodes: [out_251, out_252, out_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf253 = extern_kernels.convolution(buf251, buf252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 256, 14, 14), (50176, 196, 14, 1))
        del buf252
        buf254 = buf251; del buf251  # reuse
        # Source Nodes: [out_254, out_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf253, arg599_1, arg600_1, arg283_1, arg284_1, buf254, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg283_1
        del arg284_1
        del arg599_1
        del arg600_1
        del buf253
        # Source Nodes: [out_254, out_255, out_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf255 = extern_kernels.convolution(buf254, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg285_1
        buf256 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 0)  # alias
        buf257 = reinterpret_tensor(buf262, (8, 512, 14, 14), (551936, 1, 39424, 2816), 512)  # alias
        # Source Nodes: [cat_15, out_257, out_258, x2_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_42.run(buf255, arg602_1, arg603_1, arg286_1, arg287_1, buf249, buf256, buf257, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg286_1
        del arg287_1
        del arg602_1
        del arg603_1
        del buf249
        buf258 = reinterpret_tensor(buf262, (8, 256, 14, 14), (551936, 1, 39424, 2816), 1024)  # alias
        # Source Nodes: [bottom_8], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_43.run(buf109, buf258, 401408, grid=grid(401408), stream=stream0)
        del buf109
        del buf257
        del buf258
        del buf259
        del buf260
        del buf261
        # Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg288_1
        buf264 = reinterpret_tensor(buf255, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf255  # reuse
        # Source Nodes: [x_73, x_74, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_44.run(buf263, arg605_1, arg606_1, arg289_1, arg290_1, buf256, buf264, 1568, 512, grid=grid(1568, 512), stream=stream0)
        del arg289_1
        del arg290_1
        del arg605_1
        del arg606_1
        del buf256
        del buf262
        del buf263
        buf265 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        buf284 = empty_strided((8, 2560, 7, 7), (125440, 1, 17920, 2560), device='cuda', dtype=torch.float32)
        buf283 = reinterpret_tensor(buf284, (8, 512, 7, 7), (125440, 1, 17920, 2560), 2048)  # alias
        # Source Nodes: [bottom_23, cat_14], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_45.run(buf264, buf265, buf283, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_260], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf264, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 512, 14, 14), (100352, 196, 14, 1))
        del arg294_1
        buf267 = buf264; del buf264  # reuse
        # Source Nodes: [out_261, out_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf266, arg611_1, arg612_1, arg295_1, arg296_1, buf267, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del arg295_1
        del arg296_1
        del arg611_1
        del arg612_1
        del buf266
        buf268 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_261, out_262, out_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(arg297_1, buf268, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg297_1
        # Source Nodes: [out_261, out_262, out_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf269 = extern_kernels.convolution(buf267, buf268, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 512, 7, 7), (25088, 49, 7, 1))
        del buf267
        buf270 = empty_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_264, out_265], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf269, arg614_1, arg615_1, arg298_1, arg299_1, buf270, 4096, 49, grid=grid(4096, 49), stream=stream0)
        del arg298_1
        del arg299_1
        del arg614_1
        del arg615_1
        del buf269
        # Source Nodes: [out_264, out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf271 = extern_kernels.convolution(buf270, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf271, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg300_1
        del buf270
        # Source Nodes: [l__mod___level5_project_0], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf265, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg291_1
        buf273 = buf271; del buf271  # reuse
        buf274 = reinterpret_tensor(buf254, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf254  # reuse
        # Source Nodes: [out_267, out_268, shortcut_36, shortcut_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_49.run(buf273, arg617_1, arg618_1, arg301_1, arg302_1, buf272, arg608_1, arg609_1, arg292_1, arg293_1, buf274, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del arg292_1
        del arg293_1
        del arg301_1
        del arg302_1
        del arg608_1
        del arg609_1
        del arg617_1
        del arg618_1
        del buf272
        del buf273
        # Source Nodes: [out_270, shortcut_37], Original ATen: [aten.convolution, aten.relu]
        buf275 = extern_kernels.convolution(buf274, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 512, 7, 7), (25088, 49, 7, 1))
        del arg303_1
        buf276 = buf265; del buf265  # reuse
        # Source Nodes: [out_271, out_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf275, arg620_1, arg621_1, arg304_1, arg305_1, buf276, 4096, 49, grid=grid(4096, 49), stream=stream0)
        del arg304_1
        del arg305_1
        del arg620_1
        del arg621_1
        del buf275
        buf277 = buf268; del buf268  # reuse
        # Source Nodes: [out_271, out_272, out_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(arg306_1, buf277, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg306_1
        # Source Nodes: [out_271, out_272, out_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf278 = extern_kernels.convolution(buf276, buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 512, 7, 7), (25088, 49, 7, 1))
        del buf277
        buf279 = buf276; del buf276  # reuse
        # Source Nodes: [out_274, out_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf278, arg623_1, arg624_1, arg307_1, arg308_1, buf279, 4096, 49, grid=grid(4096, 49), stream=stream0)
        del arg307_1
        del arg308_1
        del arg623_1
        del arg624_1
        del buf278
        # Source Nodes: [out_274, out_275, out_276], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf280 = extern_kernels.convolution(buf279, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg309_1
        del buf279
        buf281 = reinterpret_tensor(buf284, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 0)  # alias
        buf282 = reinterpret_tensor(buf284, (8, 1024, 7, 7), (125440, 1, 17920, 2560), 1024)  # alias
        # Source Nodes: [cat_14, out_277, out_278, x2_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_50.run(buf280, arg626_1, arg627_1, arg310_1, arg311_1, buf274, buf281, buf282, 392, 1024, grid=grid(392, 1024), stream=stream0)
        del arg310_1
        del arg311_1
        del arg626_1
        del arg627_1
        del buf274
        del buf280
        del buf282
        del buf283
        # Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg312_1
        buf286 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf287 = reinterpret_tensor(buf286, (8, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf286  # reuse
        # Source Nodes: [x_82, x_83, x_87, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51.run(buf287, buf285, arg629_1, arg630_1, arg313_1, arg314_1, buf281, 8192, 49, grid=grid(8192), stream=stream0)
        del arg313_1
        del arg314_1
        del arg629_1
        del arg630_1
        del buf281
        del buf284
        del buf285
        # Source Nodes: [x_82, x_83, x_87, x_88, x_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.mean, aten.relu]
        buf288 = extern_kernels.convolution(buf287, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1000, 1, 1), (1000, 1, 1, 1))
        del arg315_1
        del buf287
        buf289 = reinterpret_tensor(buf288, (8, 1000), (1000, 1), 0); del buf288  # reuse
        # Source Nodes: [x_82, x_83, x_87, x_88, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.mean, aten.relu, aten.view]
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_mean_relu_view_52.run(buf289, arg316_1, 8000, grid=grid(8000), stream=stream0)
        del arg316_1
        return (buf289, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((512, 2816, 1, 1), (2816, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1000, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg320_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg323_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg326_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg329_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg332_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg338_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg341_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg344_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg347_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg350_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg353_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg359_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg362_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg365_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg368_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg371_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg374_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg380_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg383_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg386_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg389_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg392_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg395_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg398_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg401_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg404_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg407_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg410_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg413_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg416_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg419_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg422_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg425_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg428_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg431_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg434_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg437_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg440_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg443_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg446_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg449_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg452_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg455_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg458_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg461_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg464_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg467_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg470_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg473_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg476_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg482_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg485_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg488_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg491_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg494_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg500_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg503_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg506_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg512_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg518_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg521_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg524_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg527_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg530_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg533_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg536_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg539_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg542_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg545_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg548_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg551_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg554_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg557_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg560_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg563_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg566_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg569_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg572_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg575_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg578_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg581_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg584_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg587_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg590_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg593_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg596_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg599_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg602_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg605_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg608_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg611_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg614_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg617_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg620_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg623_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg626_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg629_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg632_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dla102', benchmark_compiled_module)
