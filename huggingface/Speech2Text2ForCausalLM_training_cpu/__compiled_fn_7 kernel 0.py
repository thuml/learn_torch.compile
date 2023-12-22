
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


cpp_fused__to_copy_cumsum_ne_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       int* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<long>(1);
            auto tmp2 = tmp0 != tmp1;
            auto tmp3 = c10::convert<int>(tmp2);
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


cpp_fused_detach_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp4 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = c10::convert<int>(tmp0);
                auto tmp2 = static_cast<int>(0);
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp5 = static_cast<long>(1);
                auto tmp6 = tmp4 != tmp5;
                auto tmp7 = c10::convert<int>(tmp6);
                auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                auto tmp9 = c10::convert<long>(tmp8);
                auto tmp10 = decltype(tmp9)(tmp9 + tmp5);
                auto tmp11 = decltype(tmp10)(tmp10 + 1026);
                auto tmp12 = tmp10 < 0;
                auto tmp13 = tmp12 ? tmp11 : tmp10;
                TORCH_CHECK((0 <= tmp13) & (tmp13 < 1026L), "index out of bounds: 0 <= tmp13 < 1026L")
                auto tmp14 = in_ptr2[static_cast<long>(x1 + (256L*tmp13))];
                out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp14;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1026, 256), (256, 1))
    assert_size_stride(arg1_1, (1, 128), (128, 1))
    buf0 = empty((1, 128), device='cpu', dtype=torch.int32)
    cpp_fused__to_copy_cumsum_ne_0(c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
    buf1 = aten.cumsum(buf0, 1)
    del buf0
    buf2 = buf1
    del buf1
    buf3 = empty((1, 128, 256), device='cpu', dtype=torch.float32)
    cpp_fused_detach_1(c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1026, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('Speech2Text2ForCausalLM', benchmark_compiled_module)
