
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
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<float>(0.0);
            out_ptr0[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(-100);
        auto tmp2 = tmp0 != tmp1;
        auto tmp3 = static_cast<long>(0);
        auto tmp4 = tmp2 ? tmp0 : tmp3;
        out_ptr1[static_cast<long>(0L)] = tmp4;
    }
}
''')


cpp_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        {
            float tmp_acc0 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(0L)];
                auto tmp4 = in_ptr2[static_cast<long>(0L)];
                auto tmp5 = in_ptr3[static_cast<long>(0L)];
                auto tmp2 = static_cast<long>(-100);
                auto tmp3 = tmp1 != tmp2;
                auto tmp6 = c10::convert<long>(tmp5);
                auto tmp7 = c10::convert<float>(tmp6);
                auto tmp8 = tmp4 / tmp7;
                auto tmp9 = static_cast<float>(0.0);
                auto tmp10 = tmp3 ? tmp8 : tmp9;
                auto tmp11 = decltype(tmp0)(tmp0 * tmp10);
                tmp_acc0 = tmp_acc0 + tmp11;
            }
            out_ptr0[static_cast<long>(0L)] = tmp_acc0;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(0L)];
            auto tmp4 = in_ptr2[static_cast<long>(0L)];
            auto tmp5 = in_ptr3[static_cast<long>(0L)];
            auto tmp12 = in_ptr4[static_cast<long>(x0)];
            auto tmp14 = out_ptr0[static_cast<long>(0L)];
            auto tmp2 = static_cast<long>(-100);
            auto tmp3 = tmp1 != tmp2;
            auto tmp6 = c10::convert<long>(tmp5);
            auto tmp7 = c10::convert<float>(tmp6);
            auto tmp8 = tmp4 / tmp7;
            auto tmp9 = static_cast<float>(0.0);
            auto tmp10 = tmp3 ? tmp8 : tmp9;
            auto tmp11 = decltype(tmp0)(tmp0 * tmp10);
            auto tmp13 = std::exp(tmp12);
            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
            auto tmp16 = decltype(tmp11)(tmp11 - tmp15);
            in_out_ptr0[static_cast<long>(x0)] = tmp16;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, sub_1, ne, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (1, ), (1, ))
    assert_size_stride(sub_1, (1, 2), (2, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(tangents_1, (), ())
    buf0 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 1), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf5 = buf3; del buf3  # reuse
    cpp_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(ne.data_ptr()), c_void_p(sub_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del buf4
    del ne
    del primals_2
    del sub_1
    del tangents_1
    return (buf5, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    sub_1 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, sub_1, ne, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
