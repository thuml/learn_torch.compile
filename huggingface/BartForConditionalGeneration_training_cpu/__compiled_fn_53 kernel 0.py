
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


cpp_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(51471360L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<long>(-100);
                    auto tmp2 = tmp0 != tmp1;
                    auto tmp3 = static_cast<long>(0);
                    auto tmp4 = tmp2 ? tmp0 : tmp3;
                    out_ptr1[static_cast<long>(x0)] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(0L)];
                        auto tmp5 = in_ptr3[static_cast<long>(0L)];
                        auto tmp2 = static_cast<int>(-100);
                        auto tmp3 = tmp1 != tmp2;
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(0L)];
                        auto tmp5 = in_ptr3[static_cast<long>(0L)];
                        auto tmp2 = static_cast<long>(-100);
                        auto tmp3 = tmp1 != tmp2;
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        auto tmp9 = decltype(tmp0)(tmp0 * tmp8);
                        tmp_acc0 = tmp_acc0 + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (50265L*x0)));
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<int>(-100);
                    auto tmp4 = tmp2 != tmp3;
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp4 ? tmp7 : tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp1 * tmp10;
                    auto tmp13 = tmp12.exp();
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp11 - tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp11 = in_ptr5[static_cast<long>(x1 + (50265L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<long>(-100);
                    auto tmp4 = tmp2 != tmp3;
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp4 ? tmp7 : tmp8;
                    auto tmp10 = decltype(tmp1)(tmp1 * tmp9);
                    auto tmp12 = std::exp(tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp10)(tmp10 - tmp14);
                    auto tmp16 = decltype(tmp0)(tmp0 + tmp15);
                    in_out_ptr0[static_cast<long>(x1 + (50265L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (1, 1024), (1024, 1))
    assert_size_stride(view, (1024, 1024), (1024, 1))
    assert_size_stride(sub_1, (1024, 50265), (50265, 1))
    assert_size_stride(convert_element_type, (), ())
    assert_size_stride(permute_3, (50265, 1024), (1024, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 1024, 50265), (51471360, 50265, 1))
    buf0 = empty((1024, 50265), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.float32)
    buf3 = empty((1024, 50265), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 1024, 50265), (51471360, 50265, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf4
    del convert_element_type
    del primals_4
    del sub_1
    del tangents_1
    del tangents_2
    buf6 = reinterpret_tensor(buf0, (50265, 1024), (1024, 1), 0); del buf0  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (50265, 1024), (1, 50265), 0), view, out=buf6)
    del view
    buf7 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1024, 50265), (50265, 1), 0), permute_3, out=buf7)
    del buf5
    del permute_3
    return (reinterpret_tensor(buf6, (50265, 1024), (1024, 1), 0), None, reinterpret_tensor(buf7, (1, 1024, 1024), (1048576, 1024, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    sub_1 = rand_strided((1024, 50265), (50265, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_3 = rand_strided((50265, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 1024, 50265), (51471360, 50265, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, view, sub_1, convert_element_type, permute_3, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
