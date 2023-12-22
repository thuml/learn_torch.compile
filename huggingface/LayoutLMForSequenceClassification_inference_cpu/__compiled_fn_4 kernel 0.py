
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
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
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
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp9 = out_ptr0[static_cast<long>(0L)];
        auto tmp11 = out_ptr1[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(-100);
        auto tmp2 = tmp0 != tmp1;
        auto tmp3 = static_cast<long>(0);
        auto tmp4 = tmp2 ? tmp0 : tmp3;
        auto tmp5 = decltype(tmp4)(tmp4 + 2);
        auto tmp6 = tmp4 < 0;
        auto tmp7 = tmp6 ? tmp5 : tmp4;
        TORCH_CHECK((0 <= tmp7) & (tmp7 < 2L), "index out of bounds: 0 <= tmp7 < 2L")
        auto tmp8 = in_ptr0[static_cast<long>(tmp7)];
        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
        auto tmp12 = std::log(tmp11);
        auto tmp13 = decltype(tmp10)(tmp10 - tmp12);
        auto tmp14 = decltype(tmp13)(-tmp13);
        auto tmp15 = static_cast<float>(0.0);
        auto tmp16 = tmp2 ? tmp14 : tmp15;
        auto tmp17 = c10::convert<long>(tmp2);
        auto tmp18 = c10::convert<float>(tmp17);
        auto tmp19 = tmp16 / tmp18;
        in_out_ptr0[static_cast<long>(0L)] = tmp19;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2), (2, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    buf0 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf2 = reinterpret_tensor(buf0, (), (), 0); del buf0  # reuse
    cpp_fused__log_softmax_nll_loss_forward_0(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
