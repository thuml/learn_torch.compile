
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


cpp_fused_embedding_dense_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1574400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = static_cast<int>(-1);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = to_float_mask(tmp2);
                    auto tmp6 = at::vec::Vectorized<float>(tmp4);
                    auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                    tmp7.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    add, tangents_1 = args
    args.clear()
    assert_size_stride(add, (1, 2048), (2048, 1))
    assert_size_stride(tangents_1, (1, 2048, 768), (1572864, 768, 1))
    buf0 = empty((2050, 768), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_0(c_void_p(add.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del tangents_1
    aten.index_put_(buf0, [add], buf1, True)
    del add
    del buf1
    return (buf0, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    add = rand_strided((1, 2048), (2048, 1), device='cpu', dtype=torch.int64)
    tangents_1 = rand_strided((1, 2048, 768), (1572864, 768, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([add, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
