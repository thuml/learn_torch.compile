
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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


cpp_fused__to_copy_add_embedding_mul_native_layer_norm_ne_slice_zeros_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp18 = in_ptr4[static_cast<long>(x1)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                        auto tmp4 = in_ptr1[static_cast<long>(x1 + (768L*tmp3))];
                        auto tmp6 = c10::convert<int>(tmp5);
                        auto tmp7 = static_cast<long>(1);
                        auto tmp8 = tmp0 != tmp7;
                        auto tmp9 = c10::convert<int>(tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        auto tmp11 = c10::convert<long>(tmp10);
                        auto tmp12 = decltype(tmp11)(tmp11 + tmp7);
                        auto tmp13 = decltype(tmp12)(tmp12 + 4098);
                        auto tmp14 = tmp12 < 0;
                        auto tmp15 = tmp14 ? tmp13 : tmp12;
                        TORCH_CHECK((0 <= tmp15) & (tmp15 < 4098L), "index out of bounds: 0 <= tmp15 < 4098L")
                        auto tmp16 = in_ptr3[static_cast<long>(x1 + (768L*tmp15))];
                        auto tmp17 = decltype(tmp4)(tmp4 + tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp19;
                        tmp_acc0 = welford_combine(tmp_acc0, tmp19);
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr2[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50265, 768), (768, 1))
    assert_size_stride(arg1_1, (4098, 768), (768, 1))
    assert_size_stride(arg2_1, (1, 768), (768, 1))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (1, 1024), (1024, 1))
    buf0 = empty((1, 1024), device='cpu', dtype=torch.int32)
    cpp_fused__to_copy_cumsum_ne_0(c_void_p(arg5_1.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
    buf1 = aten.cumsum(buf0, 1)
    del buf0
    buf2 = buf1
    del buf1
    buf3 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__to_copy_add_embedding_mul_native_layer_norm_ne_slice_zeros_1(c_void_p(arg5_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    return (buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50265, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((4098, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
