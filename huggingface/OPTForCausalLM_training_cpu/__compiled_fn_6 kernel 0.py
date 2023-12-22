
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


cpp_fused__to_copy_add_masked_fill_rsub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(1.0);
                    auto tmp2 = decltype(tmp1)(tmp1 - tmp0);
                    auto tmp3 = c10::convert<bool>(tmp2);
                    auto tmp4 = static_cast<float>(-3.4028234663852886e+38);
                    auto tmp5 = tmp3 ? tmp4 : tmp2;
                    auto tmp6 = c10::convert<long>(x1);
                    auto tmp7 = c10::convert<long>(1L + x0);
                    auto tmp8 = tmp6 < tmp7;
                    auto tmp9 = static_cast<float>(0.0);
                    auto tmp10 = tmp8 ? tmp9 : tmp4;
                    auto tmp11 = decltype(tmp5)(tmp5 + tmp10);
                    out_ptr0[static_cast<long>(x1 + (2048L*x0))] = tmp11;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2048), (2048, 1))
    buf0 = empty((1, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_masked_fill_rsub_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
