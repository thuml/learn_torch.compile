
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu56qnrtfkj6g7maqlmgjyuum5fnlivwp4axtkhnu3qembdcobzn.py
# Source Nodes: [l__mod___main_1, l__mod___main_2], Original ATen: [aten.convolution, aten.leaky_relu]
# l__mod___main_1 => gt, mul, where
# l__mod___main_2 => convolution_1
triton_poi_fused_convolution_leaky_relu_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/5o/c5otrr55tl427gvm26a4uqarndo7aps4hmnk6tk655yo4l2utsoy.py
# Source Nodes: [l__mod___main_3, l__mod___main_4, l__mod___main_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
# l__mod___main_3 => add_1, mul_2, mul_3, sub
# l__mod___main_4 => gt_1, mul_4, where_1
# l__mod___main_5 => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tmp17 = 0.2
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwyfxy4lpfynute4gboyjy4wjny4ermz6klrhty62hhw5ttolqvf.py
# Source Nodes: [l__mod___main_6, l__mod___main_7, l__mod___main_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
# l__mod___main_6 => add_3, mul_6, mul_7, sub_1
# l__mod___main_7 => gt_2, mul_8, where_2
# l__mod___main_8 => convolution_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 256
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
    tmp17 = 0.2
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdo4fkt2dk4s7jqtmfxusqwio4rcoirvlwcss3v7kh7uro2vpjx.py
# Source Nodes: [l__mod___main_10, l__mod___main_11, l__mod___main_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
# l__mod___main_10 => gt_3, mul_12, where_3
# l__mod___main_11 => convolution_4
# l__mod___main_9 => add_5, mul_10, mul_11, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16) % 512
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
    tmp17 = 0.2
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvvy6z72l4kfi2y4l6mggcecxfm63uxpgagjsvu7pv4nwo4nqxs.py
# Source Nodes: [l__mod___main_12], Original ATen: [aten.sigmoid]
# l__mod___main_12 => sigmoid
triton_poi_fused_sigmoid_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg1_1, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(arg5_1, (256, ), (1, ))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (1, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (), ())
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (), ())
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (), ())
    assert_size_stride(arg20_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___main_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg20_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        del arg0_1
        del arg20_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___main_1, l__mod___main_2], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_leaky_relu_0.run(buf1, 262144, grid=grid(262144), stream=stream0)
        # Source Nodes: [l__mod___main_1, l__mod___main_2], Original ATen: [aten.convolution, aten.leaky_relu]
        buf2 = extern_kernels.convolution(buf1, arg1_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 128, 16, 16), (32768, 256, 16, 1))
        del arg1_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = buf3; del buf3  # reuse
        # Source Nodes: [l__mod___main_3, l__mod___main_4, l__mod___main_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_1.run(buf4, arg11_1, arg12_1, arg2_1, arg3_1, 131072, grid=grid(131072), stream=stream0)
        del arg11_1
        del arg12_1
        del arg2_1
        del arg3_1
        # Source Nodes: [l__mod___main_4, l__mod___main_5], Original ATen: [aten.convolution, aten.leaky_relu]
        buf5 = extern_kernels.convolution(buf4, arg4_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 256, 8, 8), (16384, 64, 8, 1))
        del arg4_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [l__mod___main_6, l__mod___main_7, l__mod___main_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_2.run(buf7, arg14_1, arg15_1, arg5_1, arg6_1, 65536, grid=grid(65536), stream=stream0)
        del arg14_1
        del arg15_1
        del arg5_1
        del arg6_1
        # Source Nodes: [l__mod___main_7, l__mod___main_8], Original ATen: [aten.convolution, aten.leaky_relu]
        buf8 = extern_kernels.convolution(buf7, arg7_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 512, 4, 4), (8192, 16, 4, 1))
        del arg7_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [l__mod___main_10, l__mod___main_11, l__mod___main_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_leaky_relu_3.run(buf10, arg17_1, arg18_1, arg8_1, arg9_1, 32768, grid=grid(32768), stream=stream0)
        del arg17_1
        del arg18_1
        del arg8_1
        del arg9_1
        # Source Nodes: [l__mod___main_10, l__mod___main_11], Original ATen: [aten.convolution, aten.leaky_relu]
        buf11 = extern_kernels.convolution(buf10, arg10_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 1, 1, 1), (1, 1, 1, 1))
        del arg10_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        # Source Nodes: [l__mod___main_12], Original ATen: [aten.sigmoid]
        triton_poi_fused_sigmoid_4.run(buf12, 4, grid=grid(4), stream=stream0)
        return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg20_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dcgan', benchmark_compiled_module)
