
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


cpp_fused_add_detach_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<long>(2);
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = decltype(tmp2)(tmp2 + 2050);
                    auto tmp4 = tmp2 < 0;
                    auto tmp5 = tmp4 ? tmp3 : tmp2;
                    TORCH_CHECK((0 <= tmp5) & (tmp5 < 2050L), "index out of bounds: 0 <= tmp5 < 2050L")
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (1024L*tmp5))];
                    out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<long>(2);
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
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
    assert_size_stride(arg0_1, (2050, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1, 128), (128, 1))
    buf0 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_detach_0(c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2050, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XGLMForCausalLM', benchmark_compiled_module)
