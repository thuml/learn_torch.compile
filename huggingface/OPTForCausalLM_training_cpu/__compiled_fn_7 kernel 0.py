
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


cpp_fused__to_copy_cumsum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       long* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = c10::convert<long>(tmp0);
            out_ptr0[static_cast<long>(x0)] = tmp1;
        }
    }
}
''')


cpp_fused__to_copy_add_embedding_mul_sub_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(long* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr0[static_cast<long>(x0)];
            auto tmp2 = c10::convert<long>(tmp1);
            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
            auto tmp4 = static_cast<long>(1);
            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
            auto tmp6 = static_cast<long>(2);
            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
            in_out_ptr0[static_cast<long>(x0)] = tmp7;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 2050);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 2050L), "index out of bounds: 0 <= tmp3 < 2050L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (2050, 768), (768, 1))
    assert_size_stride(primals_2, (1, 2048), (2048, 1))
    buf0 = empty((1, 2048), device='cpu', dtype=torch.int64)
    cpp_fused__to_copy_cumsum_0(c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [attention_mask, cumsum], Original ATen: [aten._to_copy, aten.cumsum]
    buf1 = aten.cumsum(buf0, 1)
    del buf0
    buf2 = buf1
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty((1, 2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_embedding_mul_sub_1(c_void_p(buf3.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_1
    del primals_2
    return (buf4, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2050, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
