
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


cpp_fused__log_softmax_nll_loss_forward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       bool* out_ptr3,
                       float* out_ptr4)
{
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr0[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = out_ptr0[static_cast<long>(0L)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp3 = std::exp(tmp2);
                tmp_acc0 = tmp_acc0 + tmp3;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = out_ptr0[static_cast<long>(0L)];
            auto tmp3 = out_ptr1[static_cast<long>(0L)];
            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
            auto tmp4 = std::log(tmp3);
            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(-100);
        auto tmp2 = tmp0 != tmp1;
        auto tmp3 = static_cast<long>(0);
        auto tmp4 = tmp2 ? tmp0 : tmp3;
        auto tmp5 = decltype(tmp4)(tmp4 + 2);
        auto tmp6 = tmp4 < 0;
        auto tmp7 = tmp6 ? tmp5 : tmp4;
        TORCH_CHECK((0 <= tmp7) & (tmp7 < 2L), "index out of bounds: 0 <= tmp7 < 2L")
        auto tmp8 = out_ptr2[static_cast<long>(tmp7)];
        auto tmp9 = decltype(tmp8)(-tmp8);
        auto tmp10 = static_cast<float>(0.0);
        auto tmp11 = tmp2 ? tmp9 : tmp10;
        auto tmp12 = c10::convert<long>(tmp2);
        auto tmp13 = c10::convert<float>(tmp12);
        auto tmp14 = tmp11 / tmp13;
        out_ptr3[static_cast<long>(0L)] = tmp2;
        out_ptr4[static_cast<long>(0L)] = tmp14;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (1, 2), (2, 1))
    assert_size_stride(primals_2, (1, ), (1, ))
    buf0 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf2 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf3 = empty((1, ), device='cpu', dtype=torch.bool)
    buf4 = empty((), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_nll_loss_forward_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del buf1
    del primals_1
    return (buf4, primals_2, buf2, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
