
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


cpp_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1)];
                        auto tmp8 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        tmp_acc0 = tmp_acc0 + tmp7;
                        tmp_acc1 = tmp_acc1 + tmp9;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (768L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1)];
                    auto tmp11 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = in_ptr5[static_cast<long>(x0)];
                    auto tmp23 = in_ptr6[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                    auto tmp17 = decltype(tmp0)(tmp0 * tmp16);
                    auto tmp19 = static_cast<long>(1);
                    auto tmp20 = tmp18 == tmp19;
                    auto tmp21 = static_cast<float>(0.0);
                    auto tmp22 = tmp20 ? tmp21 : tmp17;
                    auto tmp24 = tmp23 == tmp19;
                    auto tmp25 = tmp24 ? tmp21 : tmp17;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp17;
                    out_ptr3[static_cast<long>(x1 + (768L*x0))] = tmp22;
                    out_ptr4[static_cast<long>(x1 + (768L*x0))] = tmp25;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (768L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (768L*x1))];
                        auto tmp6 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                        tmp_acc1 = tmp_acc1 + tmp5;
                    }
                    out_ptr5[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr6[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr2[static_cast<long>(x0)];
                auto tmp1 = static_cast<bool>(0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 ? tmp2 : tmp0;
                in_out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3147264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38603520L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_6, full_default, add, mul_2, getitem_3, div, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_6, (1, 1024), (1024, 1))
    assert_size_stride(full_default, (1, 1024), (1024, 1))
    assert_size_stride(add, (1, 1024), (1024, 1))
    assert_size_stride(mul_2, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    buf0 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf2 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf14 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf3 = empty((768, ), device='cpu', dtype=torch.float32)
    buf4 = empty((768, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf6 = buf2; del buf2  # reuse
    cpp_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_0(c_void_p(buf6.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div.data_ptr()), c_void_p(add.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del buf0
    del buf1
    del div
    del getitem_3
    del mul_2
    del primals_4
    del tangents_1
    aten.index_put_(buf5, [full_default], buf6, True)
    del buf6
    del full_default
    buf9 = empty((4098, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_1(c_void_p(buf9.data_ptr()))
    aten.index_put_(buf9, [add], buf10, True)
    del add
    del buf10
    buf13 = empty((50265, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_2(c_void_p(buf13.data_ptr()))
    aten.index_put_(buf13, [primals_6], buf14, True)
    del buf14
    del primals_6
    return (buf13, buf9, buf5, buf3, buf4, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    full_default = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    add = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    mul_2 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    div = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_6, full_default, add, mul_2, getitem_3, div, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
