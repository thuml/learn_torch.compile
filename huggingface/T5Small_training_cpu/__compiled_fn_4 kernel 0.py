
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32899072L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32128L*x0)));
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
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32128L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32128L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (512L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (512L*x1))];
                        auto tmp8 = in_ptr2[static_cast<long>(x0 + (512L*x1))];
                        auto tmp9 = in_ptr3[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.04419417382415922);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = c10::convert<float>(tmp3);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp11 = decltype(tmp7)(tmp7 * tmp10);
                        tmp_acc0 = tmp_acc0 + tmp11;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp8 = in_ptr4[static_cast<long>(x1)];
                        auto tmp10 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = static_cast<float>(0.04419417382415922);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = c10::convert<float>(tmp3);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        tmp_acc0 = tmp_acc0 + tmp11;
                    }
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = static_cast<float>(0.04419417382415922);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = c10::convert<float>(tmp3);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp10);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = decltype(tmp19)(tmp19 * tmp22);
                    auto tmp24 = decltype(tmp11)(tmp11 + tmp23);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp24;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr5[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr5[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const bool* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = in_ptr3[static_cast<long>(x0)];
                auto tmp7 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp9 = in_ptr4[static_cast<long>(x0)];
                auto tmp11 = in_ptr5[static_cast<long>(x0)];
                auto tmp13 = in_ptr6[static_cast<long>(x0)];
                auto tmp15 = in_ptr7[static_cast<long>(x0)];
                auto tmp17 = in_ptr8[static_cast<long>(x0)];
                auto tmp19 = in_ptr9[static_cast<long>(x0)];
                auto tmp21 = in_ptr10[static_cast<long>(x0)];
                auto tmp23 = in_ptr11[static_cast<long>(x0)];
                auto tmp25 = in_ptr12[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                auto tmp26 = c10::convert<float>(tmp25);
                auto tmp27 = static_cast<float>(1.1111111111111112);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp29 = decltype(tmp24)(tmp24 * tmp28);
                in_out_ptr0[static_cast<long>(x0)] = tmp29;
            }
        }
    }
}
''')


cpp_fused_mul_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_sum_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_sum_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_mul_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr1[static_cast<long>(x0)];
                auto tmp1 = in_ptr3[static_cast<long>(x0)];
                auto tmp2 = in_ptr4[static_cast<long>(x0)];
                auto tmp4 = in_ptr5[static_cast<long>(x0)];
                auto tmp6 = in_ptr6[static_cast<long>(x0)];
                auto tmp8 = in_ptr7[static_cast<long>(x0)];
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = decltype(tmp9)(tmp9 + tmp0);
                auto tmp11 = static_cast<bool>(0);
                auto tmp12 = static_cast<float>(0.0);
                auto tmp13 = tmp11 ? tmp12 : tmp10;
                out_ptr2[static_cast<long>(x0)] = tmp0;
                out_ptr3[static_cast<long>(x0)] = tmp13;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (1024L*x2) + (65536L*x0)), static_cast<long>(1024L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (65536L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (512L*x1) + (512L*x1_inner)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const long* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr5[static_cast<long>(x1)];
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp19 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr6[static_cast<long>(x1 + (512L*x0))];
                    auto tmp29 = in_ptr7[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                    auto tmp12 = static_cast<float>(-0.5);
                    auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                    auto tmp14 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp8);
                    auto tmp16 = decltype(tmp13)(tmp13 * tmp15);
                    auto tmp17 = static_cast<float>(512.0);
                    auto tmp18 = tmp16 / tmp17;
                    auto tmp20 = static_cast<float>(2.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp18)(tmp18 * tmp21);
                    auto tmp23 = decltype(tmp10)(tmp10 + tmp22);
                    auto tmp25 = c10::convert<float>(tmp24);
                    auto tmp26 = static_cast<float>(1.1111111111111112);
                    auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                    auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                    auto tmp30 = static_cast<long>(-1);
                    auto tmp31 = tmp29 == tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp31 ? tmp32 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16449536L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = in_ptr2[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    auto tmp7 = static_cast<float>(-0.5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = decltype(tmp3)(tmp3 * tmp3);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp3);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = static_cast<float>(512.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = static_cast<float>(2.0);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp13);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp5 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp0 + tmp10;
                    auto tmp13 = static_cast<float>(-0.5);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    auto tmp18 = static_cast<float>(512.0);
                    auto tmp19 = tmp17 / tmp18;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp19);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp11 + tmp25;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp2 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp0 ? tmp7 : tmp6;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp9 = static_cast<float>(-0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = decltype(tmp4)(tmp4 * tmp4);
                    auto tmp12 = decltype(tmp11)(tmp11 * tmp4);
                    auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                    auto tmp14 = static_cast<float>(512.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp17 = static_cast<float>(2.0);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp15);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp7 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp10;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr1[static_cast<long>(x0)];
                auto tmp1 = in_ptr3[static_cast<long>(x0)];
                auto tmp2 = in_ptr4[static_cast<long>(x0)];
                auto tmp4 = in_ptr5[static_cast<long>(x0)];
                auto tmp6 = in_ptr6[static_cast<long>(x0)];
                auto tmp8 = in_ptr7[static_cast<long>(x0)];
                auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                auto tmp10 = decltype(tmp9)(tmp9 + tmp0);
                auto tmp11 = static_cast<bool>(0);
                auto tmp12 = static_cast<float>(0.0);
                auto tmp13 = tmp11 ? tmp12 : tmp10;
                out_ptr2[static_cast<long>(x0)] = tmp0;
                out_ptr3[static_cast<long>(x0)] = tmp13;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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


cpp_fused_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       const long* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr5[static_cast<long>(x1)];
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp19 = in_ptr3[static_cast<long>(x1 + (512L*x0))];
                    auto tmp24 = in_ptr6[static_cast<long>(x1 + (512L*x0))];
                    auto tmp29 = in_ptr7[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp0)(tmp0 + tmp9);
                    auto tmp12 = static_cast<float>(-0.5);
                    auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                    auto tmp14 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp8);
                    auto tmp16 = decltype(tmp13)(tmp13 * tmp15);
                    auto tmp17 = static_cast<float>(512.0);
                    auto tmp18 = tmp16 / tmp17;
                    auto tmp20 = static_cast<float>(2.0);
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp18)(tmp18 * tmp21);
                    auto tmp23 = decltype(tmp10)(tmp10 + tmp22);
                    auto tmp25 = c10::convert<float>(tmp24);
                    auto tmp26 = static_cast<float>(1.1111111111111112);
                    auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                    auto tmp28 = decltype(tmp23)(tmp23 * tmp27);
                    auto tmp30 = static_cast<long>(-1);
                    auto tmp31 = tmp29 == tmp30;
                    auto tmp32 = static_cast<float>(0.0);
                    auto tmp33 = tmp31 ? tmp32 : tmp28;
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp33;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16449536L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16449536L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, getitem_7, view_23, getitem_9, add_8, rsqrt_2, view_25, getitem_11, view_43, getitem_13, add_11, rsqrt_3, view_45, getitem_15, view_47, getitem_17, add_13, rsqrt_4, view_49, getitem_19, view_67, getitem_21, add_16, rsqrt_5, view_69, getitem_23, view_71, getitem_25, add_18, rsqrt_6, view_73, getitem_27, view_91, getitem_29, add_21, rsqrt_7, view_93, getitem_31, view_95, getitem_33, add_23, rsqrt_8, view_97, getitem_35, view_115, getitem_37, add_26, rsqrt_9, view_117, getitem_39, view_119, getitem_41, add_28, rsqrt_10, view_121, getitem_43, view_139, getitem_45, add_31, rsqrt_11, view_141, getitem_47, view_143, getitem_49, add_33, rsqrt_12, getitem_51, view_145, getitem_52, getitem_53, rsqrt_13, view_146, add_37, getitem_55, view_164, getitem_57, add_40, rsqrt_14, view_166, view_169, getitem_59, view_184, getitem_61, add_44, rsqrt_15, view_186, getitem_63, view_188, getitem_65, add_46, rsqrt_16, view_190, getitem_67, view_208, getitem_69, add_49, rsqrt_17, view_210, getitem_71, view_228, getitem_73, add_52, rsqrt_18, view_230, getitem_75, view_232, getitem_77, add_54, rsqrt_19, view_234, getitem_79, view_252, getitem_81, add_57, rsqrt_20, view_254, getitem_83, view_272, getitem_85, add_60, rsqrt_21, view_274, getitem_87, view_276, getitem_89, add_62, rsqrt_22, view_278, getitem_91, view_296, getitem_93, add_65, rsqrt_23, view_298, getitem_95, view_316, getitem_97, add_68, rsqrt_24, view_318, getitem_99, view_320, getitem_101, add_70, rsqrt_25, view_322, getitem_103, view_340, getitem_105, add_73, rsqrt_26, view_342, getitem_107, view_360, getitem_109, add_76, rsqrt_27, view_362, getitem_111, view_364, getitem_113, add_78, rsqrt_28, view_366, getitem_115, view_384, getitem_117, add_81, rsqrt_29, view_386, getitem_119, view_404, getitem_121, add_84, rsqrt_30, view_406, getitem_123, view_408, getitem_125, add_86, rsqrt_31, getitem_127, view_410, sub_24, convert_element_type_7, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_67, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_69, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_73, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_75, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_79, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_81, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_85, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_87, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_91, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_93, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_97, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_99, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_104, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_108, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_112, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_116, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_120, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_124, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27 = args
    args.clear()
    assert_size_stride(primals_1, (512, ), (1, ))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_134, (1, 1024), (1024, 1))
    assert_size_stride(view, (1, 1024), (1024, 1))
    assert_size_stride(getitem, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(getitem_1, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_1, (1024, 512), (512, 1))
    assert_size_stride(add_3, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_3, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_19, (1024, 512), (512, 1))
    assert_size_stride(getitem_5, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_6, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_1, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_21, (1024, 512), (512, 1))
    assert_size_stride(getitem_7, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_23, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_9, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_8, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_2, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_25, (1024, 512), (512, 1))
    assert_size_stride(getitem_11, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_43, (1024, 512), (512, 1))
    assert_size_stride(getitem_13, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_11, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_3, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_45, (1024, 512), (512, 1))
    assert_size_stride(getitem_15, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_47, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_17, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_13, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_4, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_49, (1024, 512), (512, 1))
    assert_size_stride(getitem_19, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_67, (1024, 512), (512, 1))
    assert_size_stride(getitem_21, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_16, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_5, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_69, (1024, 512), (512, 1))
    assert_size_stride(getitem_23, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_71, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_25, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_18, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_6, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_73, (1024, 512), (512, 1))
    assert_size_stride(getitem_27, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_91, (1024, 512), (512, 1))
    assert_size_stride(getitem_29, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_21, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_7, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_93, (1024, 512), (512, 1))
    assert_size_stride(getitem_31, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_95, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_33, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_23, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_8, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_97, (1024, 512), (512, 1))
    assert_size_stride(getitem_35, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_115, (1024, 512), (512, 1))
    assert_size_stride(getitem_37, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_26, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_9, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_117, (1024, 512), (512, 1))
    assert_size_stride(getitem_39, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_119, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_41, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_28, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_10, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_121, (1024, 512), (512, 1))
    assert_size_stride(getitem_43, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_139, (1024, 512), (512, 1))
    assert_size_stride(getitem_45, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_31, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_11, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_141, (1024, 512), (512, 1))
    assert_size_stride(getitem_47, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_143, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_49, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_33, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_12, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(getitem_51, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(view_145, (1, 1024), (1024, 1))
    assert_size_stride(getitem_52, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(getitem_53, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_13, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_146, (1024, 512), (512, 1))
    assert_size_stride(add_37, (1024, 1024), (1024, 1))
    assert_size_stride(getitem_55, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_164, (1024, 512), (512, 1))
    assert_size_stride(getitem_57, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_40, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_14, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_166, (1024, 512), (512, 1))
    assert_size_stride(view_169, (1024, 512), (512, 1))
    assert_size_stride(getitem_59, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_184, (1024, 512), (512, 1))
    assert_size_stride(getitem_61, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_44, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_15, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_186, (1024, 512), (512, 1))
    assert_size_stride(getitem_63, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_188, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_65, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_46, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_16, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_190, (1024, 512), (512, 1))
    assert_size_stride(getitem_67, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_208, (1024, 512), (512, 1))
    assert_size_stride(getitem_69, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_49, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_17, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_210, (1024, 512), (512, 1))
    assert_size_stride(getitem_71, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_228, (1024, 512), (512, 1))
    assert_size_stride(getitem_73, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_52, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_18, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_230, (1024, 512), (512, 1))
    assert_size_stride(getitem_75, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_232, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_77, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_54, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_19, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_234, (1024, 512), (512, 1))
    assert_size_stride(getitem_79, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_252, (1024, 512), (512, 1))
    assert_size_stride(getitem_81, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_57, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_20, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_254, (1024, 512), (512, 1))
    assert_size_stride(getitem_83, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_272, (1024, 512), (512, 1))
    assert_size_stride(getitem_85, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_60, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_21, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_274, (1024, 512), (512, 1))
    assert_size_stride(getitem_87, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_276, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_89, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_62, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_22, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_278, (1024, 512), (512, 1))
    assert_size_stride(getitem_91, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_296, (1024, 512), (512, 1))
    assert_size_stride(getitem_93, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_65, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_23, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_298, (1024, 512), (512, 1))
    assert_size_stride(getitem_95, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_316, (1024, 512), (512, 1))
    assert_size_stride(getitem_97, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_68, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_24, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_318, (1024, 512), (512, 1))
    assert_size_stride(getitem_99, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_320, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_101, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_70, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_25, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_322, (1024, 512), (512, 1))
    assert_size_stride(getitem_103, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_340, (1024, 512), (512, 1))
    assert_size_stride(getitem_105, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_73, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_26, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_342, (1024, 512), (512, 1))
    assert_size_stride(getitem_107, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_360, (1024, 512), (512, 1))
    assert_size_stride(getitem_109, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_76, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_27, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_362, (1024, 512), (512, 1))
    assert_size_stride(getitem_111, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_364, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_113, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_78, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_28, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_366, (1024, 512), (512, 1))
    assert_size_stride(getitem_115, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_384, (1024, 512), (512, 1))
    assert_size_stride(getitem_117, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_81, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_29, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_386, (1024, 512), (512, 1))
    assert_size_stride(getitem_119, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(view_404, (1024, 512), (512, 1))
    assert_size_stride(getitem_121, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_84, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_30, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_406, (1024, 512), (512, 1))
    assert_size_stride(getitem_123, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(view_408, (1024, 2048), (2048, 1))
    assert_size_stride(getitem_125, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(add_86, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_31, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(getitem_127, (1, 1024, 512), (524288, 512, 1))
    assert_size_stride(view_410, (1024, 512), (512, 1))
    assert_size_stride(sub_24, (1024, 32128), (32128, 1))
    assert_size_stride(convert_element_type_7, (), ())
    assert_size_stride(permute_191, (32128, 512), (512, 1))
    assert_size_stride(permute_195, (512, 2048), (2048, 1))
    assert_size_stride(le_1, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_199, (2048, 512), (512, 1))
    assert_size_stride(permute_203, (512, 512), (512, 1))
    assert_size_stride(permute_206, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_207, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_67, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_208, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_209, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_214, (512, 512), (512, 1))
    assert_size_stride(permute_219, (512, 512), (512, 1))
    assert_size_stride(permute_224, (512, 512), (512, 1))
    assert_size_stride(permute_228, (512, 512), (512, 1))
    assert_size_stride(permute_231, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_232, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_69, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_233, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_234, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_239, (512, 512), (512, 1))
    assert_size_stride(permute_244, (512, 512), (512, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(permute_253, (512, 2048), (2048, 1))
    assert_size_stride(le_2, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_257, (2048, 512), (512, 1))
    assert_size_stride(permute_261, (512, 512), (512, 1))
    assert_size_stride(permute_264, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_265, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_73, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_266, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_267, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_272, (512, 512), (512, 1))
    assert_size_stride(permute_277, (512, 512), (512, 1))
    assert_size_stride(permute_282, (512, 512), (512, 1))
    assert_size_stride(permute_286, (512, 512), (512, 1))
    assert_size_stride(permute_289, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_290, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_75, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_291, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_292, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_297, (512, 512), (512, 1))
    assert_size_stride(permute_302, (512, 512), (512, 1))
    assert_size_stride(permute_307, (512, 512), (512, 1))
    assert_size_stride(permute_311, (512, 2048), (2048, 1))
    assert_size_stride(le_3, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_315, (2048, 512), (512, 1))
    assert_size_stride(permute_319, (512, 512), (512, 1))
    assert_size_stride(permute_322, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_323, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_79, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_324, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_325, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_330, (512, 512), (512, 1))
    assert_size_stride(permute_335, (512, 512), (512, 1))
    assert_size_stride(permute_340, (512, 512), (512, 1))
    assert_size_stride(permute_344, (512, 512), (512, 1))
    assert_size_stride(permute_347, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_348, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_81, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_349, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_350, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_355, (512, 512), (512, 1))
    assert_size_stride(permute_360, (512, 512), (512, 1))
    assert_size_stride(permute_365, (512, 512), (512, 1))
    assert_size_stride(permute_369, (512, 2048), (2048, 1))
    assert_size_stride(le_4, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_373, (2048, 512), (512, 1))
    assert_size_stride(permute_377, (512, 512), (512, 1))
    assert_size_stride(permute_380, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_381, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_85, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_382, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_383, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_388, (512, 512), (512, 1))
    assert_size_stride(permute_393, (512, 512), (512, 1))
    assert_size_stride(permute_398, (512, 512), (512, 1))
    assert_size_stride(permute_402, (512, 512), (512, 1))
    assert_size_stride(permute_405, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_406, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_87, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_407, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_408, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_413, (512, 512), (512, 1))
    assert_size_stride(permute_418, (512, 512), (512, 1))
    assert_size_stride(permute_423, (512, 512), (512, 1))
    assert_size_stride(permute_427, (512, 2048), (2048, 1))
    assert_size_stride(le_5, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_431, (2048, 512), (512, 1))
    assert_size_stride(permute_435, (512, 512), (512, 1))
    assert_size_stride(permute_438, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_439, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_91, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_440, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_441, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_446, (512, 512), (512, 1))
    assert_size_stride(permute_451, (512, 512), (512, 1))
    assert_size_stride(permute_456, (512, 512), (512, 1))
    assert_size_stride(permute_460, (512, 512), (512, 1))
    assert_size_stride(permute_463, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_464, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_93, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_465, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_466, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_471, (512, 512), (512, 1))
    assert_size_stride(permute_476, (512, 512), (512, 1))
    assert_size_stride(permute_481, (512, 512), (512, 1))
    assert_size_stride(permute_485, (512, 2048), (2048, 1))
    assert_size_stride(le_6, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_489, (2048, 512), (512, 1))
    assert_size_stride(permute_493, (512, 512), (512, 1))
    assert_size_stride(permute_496, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_497, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_97, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_498, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_499, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_504, (512, 512), (512, 1))
    assert_size_stride(permute_509, (512, 512), (512, 1))
    assert_size_stride(permute_514, (512, 512), (512, 1))
    assert_size_stride(permute_518, (512, 512), (512, 1))
    assert_size_stride(permute_521, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_522, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_99, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_524, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_525, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_530, (512, 512), (512, 1))
    assert_size_stride(permute_535, (512, 512), (512, 1))
    assert_size_stride(permute_540, (512, 512), (512, 1))
    assert_size_stride(permute_544, (512, 2048), (2048, 1))
    assert_size_stride(le_7, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_548, (2048, 512), (512, 1))
    assert_size_stride(permute_552, (512, 512), (512, 1))
    assert_size_stride(permute_555, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_556, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_104, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_557, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_558, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_563, (512, 512), (512, 1))
    assert_size_stride(permute_568, (512, 512), (512, 1))
    assert_size_stride(permute_573, (512, 512), (512, 1))
    assert_size_stride(permute_577, (512, 2048), (2048, 1))
    assert_size_stride(le_8, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_581, (2048, 512), (512, 1))
    assert_size_stride(permute_585, (512, 512), (512, 1))
    assert_size_stride(permute_588, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_589, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_108, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_590, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_591, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_596, (512, 512), (512, 1))
    assert_size_stride(permute_601, (512, 512), (512, 1))
    assert_size_stride(permute_606, (512, 512), (512, 1))
    assert_size_stride(permute_610, (512, 2048), (2048, 1))
    assert_size_stride(le_9, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_614, (2048, 512), (512, 1))
    assert_size_stride(permute_618, (512, 512), (512, 1))
    assert_size_stride(permute_621, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_622, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_112, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_623, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_624, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_629, (512, 512), (512, 1))
    assert_size_stride(permute_634, (512, 512), (512, 1))
    assert_size_stride(permute_639, (512, 512), (512, 1))
    assert_size_stride(permute_643, (512, 2048), (2048, 1))
    assert_size_stride(le_10, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_647, (2048, 512), (512, 1))
    assert_size_stride(permute_651, (512, 512), (512, 1))
    assert_size_stride(permute_654, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_655, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_116, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_656, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_657, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_662, (512, 512), (512, 1))
    assert_size_stride(permute_667, (512, 512), (512, 1))
    assert_size_stride(permute_672, (512, 512), (512, 1))
    assert_size_stride(permute_676, (512, 2048), (2048, 1))
    assert_size_stride(le_11, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_680, (2048, 512), (512, 1))
    assert_size_stride(permute_684, (512, 512), (512, 1))
    assert_size_stride(permute_687, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_688, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_120, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_689, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_690, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_695, (512, 512), (512, 1))
    assert_size_stride(permute_700, (512, 512), (512, 1))
    assert_size_stride(permute_705, (512, 512), (512, 1))
    assert_size_stride(permute_709, (512, 2048), (2048, 1))
    assert_size_stride(le_12, (1, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_713, (2048, 512), (512, 1))
    assert_size_stride(permute_717, (512, 512), (512, 1))
    assert_size_stride(permute_720, (8, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_721, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(alias_124, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_723, (8, 64, 1024), (64, 1, 512))
    assert_size_stride(permute_724, (8, 1024, 64), (64, 512, 1))
    assert_size_stride(permute_729, (512, 512), (512, 1))
    assert_size_stride(permute_734, (512, 512), (512, 1))
    assert_size_stride(permute_739, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 1024, 32128), (32899072, 32128, 1))
    assert_size_stride(tangents_3, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_4, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_5, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_6, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_7, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_8, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_9, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_10, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_11, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_12, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_13, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_14, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_15, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_16, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_17, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_18, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_19, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_20, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_21, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_22, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_23, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_24, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_25, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_26, (1, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_27, (1, 1024, 512), (524288, 512, 1))
    buf0 = empty((1024, 32128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_134.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((1024, 1), (1, 1024), device='cpu', dtype=torch.float32)
    buf3 = empty((1024, 32128), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 1024, 32128), (32899072, 32128, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type_7.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_24.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type_7
    del primals_134
    del sub_24
    del tangents_1
    del tangents_2
    buf6 = empty((32128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (32128, 1024), (1, 32128), 0), view_410, out=buf6)
    del view_410
    buf7 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1024, 32128), (32128, 1), 0), permute_191, out=buf7)
    del buf5
    del permute_191
    buf8 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 1024, 1), (1024, 1, 1024), 0); del buf4  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 1024, 512), (524288, 512, 1), 0); del buf7  # reuse
    buf11 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_2(c_void_p(buf10.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(add_86.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(getitem_125.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    del add_86
    del getitem_125
    del getitem_127
    del primals_32
    del rsqrt_31
    buf12 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (512, 1024), (1, 512), 0), view_408, out=buf12)
    del view_408
    buf13 = empty((1024, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1024, 512), (512, 1), 0), permute_195, out=buf13)
    del permute_195
    buf14 = reinterpret_tensor(buf13, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf13  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_3(c_void_p(buf14.data_ptr()), c_void_p(le_1.data_ptr()), c_void_p(getitem_123.data_ptr()))
    del getitem_123
    del le_1
    buf15 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (2048, 1024), (1, 2048), 0), view_406, out=buf15)
    del view_406
    buf16 = reinterpret_tensor(buf11, (1024, 512), (512, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (1024, 2048), (2048, 1), 0), permute_199, out=buf16)
    del permute_199
    buf17 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf18 = buf9; del buf9  # reuse
    buf19 = buf10; del buf10  # reuse
    buf20 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_4(c_void_p(buf19.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(add_84.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del add_84
    del getitem_121
    del primals_31
    del rsqrt_30
    buf21 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (512, 1024), (1, 512), 0), view_404, out=buf21)
    del view_404
    buf22 = buf16; del buf16  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (1024, 512), (512, 1), 0), permute_203, out=buf22)
    del permute_203
    buf23 = reinterpret_tensor(buf20, (8, 1024, 64), (65536, 64, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_206, reinterpret_tensor(buf22, (8, 1024, 64), (64, 512, 1), 0), out=buf23)
    del permute_206
    buf24 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (8, 1024, 64), (64, 512, 1), 0), permute_207, out=buf24)
    del permute_207
    buf25 = empty_strided((1, 8, 1024, 1), (8192, 1024, 1, 8192), device='cpu', dtype=torch.float32)
    buf26 = buf24; del buf24  # reuse
    buf28 = empty((8388608, ), device='cpu', dtype=torch.float32)
    buf31 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_5(c_void_p(buf26.data_ptr()), c_void_p(getitem_119.data_ptr()), c_void_p(alias_67.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf31.data_ptr()))
    del alias_67
    del getitem_119
    buf33 = reinterpret_tensor(buf22, (8, 64, 1024), (65536, 1024, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_208, buf31, out=buf33)
    del permute_208
    buf34 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf31, permute_209, out=buf34)
    del permute_209
    buf35 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(tangents_26.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf35.data_ptr()))
    del tangents_26
    buf36 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (512, 1024), (1, 512), 0), view_169, out=buf36)
    buf37 = reinterpret_tensor(buf23, (1024, 512), (512, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (1024, 512), (512, 1), 0), permute_214, out=buf37)
    del permute_214
    buf38 = buf35; del buf35  # reuse
    cpp_fused_clone_7(c_void_p(tangents_25.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf38.data_ptr()))
    del tangents_25
    buf39 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (512, 1024), (1, 512), 0), view_169, out=buf39)
    buf40 = reinterpret_tensor(buf33, (1024, 512), (512, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (1024, 512), (512, 1), 0), permute_219, out=buf40)
    del permute_219
    buf41 = reinterpret_tensor(buf38, (1024, 512), (512, 1), 0); del buf38  # reuse
    cpp_fused_view_8(c_void_p(buf34.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (512, 1024), (1, 512), 0), view_386, out=buf42)
    del view_386
    buf43 = reinterpret_tensor(buf34, (1024, 512), (512, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf41, permute_224, out=buf43)
    del permute_224
    buf44 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf45 = buf18; del buf18  # reuse
    buf46 = buf19; del buf19  # reuse
    buf47 = reinterpret_tensor(buf41, (1, 1024, 512), (524288, 512, 1), 0); del buf41  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_9(c_void_p(buf46.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(add_81.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    del add_81
    del getitem_117
    del primals_30
    del rsqrt_29
    buf48 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (512, 1024), (1, 512), 0), view_384, out=buf48)
    del view_384
    buf49 = buf43; del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (1024, 512), (512, 1), 0), permute_228, out=buf49)
    del permute_228
    buf50 = reinterpret_tensor(buf47, (8, 1024, 64), (65536, 64, 1), 0); del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_231, reinterpret_tensor(buf49, (8, 1024, 64), (64, 512, 1), 0), out=buf50)
    del permute_231
    buf51 = buf31; del buf31  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (8, 1024, 64), (64, 512, 1), 0), permute_232, out=buf51)
    del permute_232
    buf52 = buf25; del buf25  # reuse
    buf53 = buf51; del buf51  # reuse
    buf54 = buf28; del buf28  # reuse
    buf57 = buf26; del buf26  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_10(c_void_p(buf53.data_ptr()), c_void_p(getitem_115.data_ptr()), c_void_p(alias_69.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf57.data_ptr()))
    del alias_69
    del getitem_115
    buf59 = reinterpret_tensor(buf49, (8, 64, 1024), (65536, 1024, 1), 0); del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_233, buf57, out=buf59)
    del permute_233
    buf60 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf57, permute_234, out=buf60)
    del permute_234
    buf61 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_11(c_void_p(tangents_24.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf61.data_ptr()))
    del tangents_24
    buf62 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (512, 1024), (1, 512), 0), view_366, out=buf62)
    buf63 = reinterpret_tensor(buf50, (1024, 512), (512, 1), 0); del buf50  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (1024, 512), (512, 1), 0), permute_239, out=buf63)
    del permute_239
    buf64 = buf61; del buf61  # reuse
    cpp_fused_clone_12(c_void_p(tangents_23.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf64.data_ptr()))
    del tangents_23
    buf65 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (512, 1024), (1, 512), 0), view_366, out=buf65)
    buf66 = reinterpret_tensor(buf59, (1024, 512), (512, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (1024, 512), (512, 1), 0), permute_244, out=buf66)
    del permute_244
    buf67 = reinterpret_tensor(buf64, (1024, 512), (512, 1), 0); del buf64  # reuse
    cpp_fused_view_13(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (512, 1024), (1, 512), 0), view_366, out=buf68)
    del view_366
    buf69 = reinterpret_tensor(buf60, (1024, 512), (512, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf67, permute_249, out=buf69)
    del permute_249
    buf70 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf71 = buf45; del buf45  # reuse
    buf72 = buf46; del buf46  # reuse
    buf73 = reinterpret_tensor(buf67, (1, 1024, 512), (524288, 512, 1), 0); del buf67  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_14(c_void_p(buf72.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(add_78.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(getitem_113.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del add_78
    del getitem_113
    del primals_29
    del rsqrt_28
    buf74 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (512, 1024), (1, 512), 0), view_364, out=buf74)
    del view_364
    buf75 = reinterpret_tensor(buf14, (1024, 2048), (2048, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (1024, 512), (512, 1), 0), permute_253, out=buf75)
    del permute_253
    buf76 = reinterpret_tensor(buf75, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf75  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_15(c_void_p(buf76.data_ptr()), c_void_p(le_2.data_ptr()), c_void_p(getitem_111.data_ptr()))
    del getitem_111
    del le_2
    buf77 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (2048, 1024), (1, 2048), 0), view_362, out=buf77)
    del view_362
    buf78 = reinterpret_tensor(buf73, (1024, 512), (512, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (1024, 2048), (2048, 1), 0), permute_257, out=buf78)
    del permute_257
    buf79 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf80 = buf71; del buf71  # reuse
    buf81 = buf72; del buf72  # reuse
    buf82 = reinterpret_tensor(buf69, (1, 1024, 512), (524288, 512, 1), 0); del buf69  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_16(c_void_p(buf81.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(add_76.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(getitem_109.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del add_76
    del getitem_109
    del primals_28
    del rsqrt_27
    buf83 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (512, 1024), (1, 512), 0), view_360, out=buf83)
    del view_360
    buf84 = buf78; del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf82, (1024, 512), (512, 1), 0), permute_261, out=buf84)
    del permute_261
    buf85 = reinterpret_tensor(buf82, (8, 1024, 64), (65536, 64, 1), 0); del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_264, reinterpret_tensor(buf84, (8, 1024, 64), (64, 512, 1), 0), out=buf85)
    del permute_264
    buf86 = buf57; del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (8, 1024, 64), (64, 512, 1), 0), permute_265, out=buf86)
    del permute_265
    buf87 = buf52; del buf52  # reuse
    buf88 = buf86; del buf86  # reuse
    buf89 = reinterpret_tensor(buf53, (8388608, ), (1, ), 0); del buf53  # reuse
    buf92 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_17(c_void_p(buf88.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(alias_73.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf92.data_ptr()))
    del alias_73
    del getitem_107
    buf94 = reinterpret_tensor(buf84, (8, 64, 1024), (65536, 1024, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_266, buf92, out=buf94)
    del permute_266
    buf95 = reinterpret_tensor(buf66, (8, 1024, 64), (65536, 64, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf92, permute_267, out=buf95)
    del permute_267
    buf96 = reinterpret_tensor(buf63, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf63  # reuse
    cpp_fused_clone_18(c_void_p(tangents_22.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf96.data_ptr()))
    del tangents_22
    buf97 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (512, 1024), (1, 512), 0), view_169, out=buf97)
    buf98 = reinterpret_tensor(buf85, (1024, 512), (512, 1), 0); del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (1024, 512), (512, 1), 0), permute_272, out=buf98)
    del permute_272
    buf99 = buf96; del buf96  # reuse
    cpp_fused_clone_19(c_void_p(tangents_21.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf99.data_ptr()))
    del tangents_21
    buf100 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (512, 1024), (1, 512), 0), view_169, out=buf100)
    buf101 = reinterpret_tensor(buf94, (1024, 512), (512, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (1024, 512), (512, 1), 0), permute_277, out=buf101)
    del permute_277
    buf102 = reinterpret_tensor(buf99, (1024, 512), (512, 1), 0); del buf99  # reuse
    cpp_fused_view_20(c_void_p(buf95.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (512, 1024), (1, 512), 0), view_342, out=buf103)
    del view_342
    buf104 = reinterpret_tensor(buf95, (1024, 512), (512, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf102, permute_282, out=buf104)
    del permute_282
    buf105 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf106 = buf80; del buf80  # reuse
    buf107 = reinterpret_tensor(buf104, (1, 1024, 512), (524288, 512, 1), 0); del buf104  # reuse
    buf108 = reinterpret_tensor(buf102, (1, 1024, 512), (524288, 512, 1), 0); del buf102  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_21(c_void_p(buf107.data_ptr()), c_void_p(add_73.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(getitem_105.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()))
    del add_73
    del getitem_105
    del primals_27
    del rsqrt_26
    buf109 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf108, (512, 1024), (1, 512), 0), view_340, out=buf109)
    del view_340
    buf110 = reinterpret_tensor(buf81, (1024, 512), (512, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf108, (1024, 512), (512, 1), 0), permute_286, out=buf110)
    del permute_286
    buf111 = reinterpret_tensor(buf108, (8, 1024, 64), (65536, 64, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_289, reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), out=buf111)
    del permute_289
    buf112 = buf92; del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (8, 1024, 64), (64, 512, 1), 0), permute_290, out=buf112)
    del permute_290
    buf113 = buf87; del buf87  # reuse
    buf114 = buf112; del buf112  # reuse
    buf115 = buf89; del buf89  # reuse
    buf118 = buf88; del buf88  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_22(c_void_p(buf114.data_ptr()), c_void_p(getitem_103.data_ptr()), c_void_p(alias_75.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()))
    del alias_75
    del getitem_103
    buf120 = reinterpret_tensor(buf110, (8, 64, 1024), (65536, 1024, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_291, buf118, out=buf120)
    del permute_291
    buf121 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf118, permute_292, out=buf121)
    del permute_292
    buf122 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_23(c_void_p(tangents_20.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf122.data_ptr()))
    del tangents_20
    buf123 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (512, 1024), (1, 512), 0), view_322, out=buf123)
    buf124 = reinterpret_tensor(buf111, (1024, 512), (512, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (1024, 512), (512, 1), 0), permute_297, out=buf124)
    del permute_297
    buf125 = buf122; del buf122  # reuse
    cpp_fused_clone_24(c_void_p(tangents_19.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf125.data_ptr()))
    del tangents_19
    buf126 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (512, 1024), (1, 512), 0), view_322, out=buf126)
    buf127 = reinterpret_tensor(buf120, (1024, 512), (512, 1), 0); del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (1024, 512), (512, 1), 0), permute_302, out=buf127)
    del permute_302
    buf128 = reinterpret_tensor(buf125, (1024, 512), (512, 1), 0); del buf125  # reuse
    cpp_fused_view_25(c_void_p(buf121.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (512, 1024), (1, 512), 0), view_322, out=buf129)
    del view_322
    buf130 = reinterpret_tensor(buf121, (1024, 512), (512, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf128, permute_307, out=buf130)
    del permute_307
    buf131 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf132 = buf106; del buf106  # reuse
    buf133 = buf107; del buf107  # reuse
    buf134 = reinterpret_tensor(buf128, (1, 1024, 512), (524288, 512, 1), 0); del buf128  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_26(c_void_p(buf133.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()))
    del add_70
    del getitem_101
    del primals_26
    del rsqrt_25
    buf135 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (512, 1024), (1, 512), 0), view_320, out=buf135)
    del view_320
    buf136 = reinterpret_tensor(buf76, (1024, 2048), (2048, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (1024, 512), (512, 1), 0), permute_311, out=buf136)
    del permute_311
    buf137 = reinterpret_tensor(buf136, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf136  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_27(c_void_p(buf137.data_ptr()), c_void_p(le_3.data_ptr()), c_void_p(getitem_99.data_ptr()))
    del getitem_99
    del le_3
    buf138 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (2048, 1024), (1, 2048), 0), view_318, out=buf138)
    del view_318
    buf139 = reinterpret_tensor(buf134, (1024, 512), (512, 1), 0); del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (1024, 2048), (2048, 1), 0), permute_315, out=buf139)
    del permute_315
    buf140 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf141 = buf132; del buf132  # reuse
    buf142 = buf133; del buf133  # reuse
    buf143 = reinterpret_tensor(buf130, (1, 1024, 512), (524288, 512, 1), 0); del buf130  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_28(c_void_p(buf142.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(add_68.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del add_68
    del getitem_97
    del primals_25
    del rsqrt_24
    buf144 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf143, (512, 1024), (1, 512), 0), view_316, out=buf144)
    del view_316
    buf145 = buf139; del buf139  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf143, (1024, 512), (512, 1), 0), permute_319, out=buf145)
    del permute_319
    buf146 = reinterpret_tensor(buf143, (8, 1024, 64), (65536, 64, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_322, reinterpret_tensor(buf145, (8, 1024, 64), (64, 512, 1), 0), out=buf146)
    del permute_322
    buf147 = buf118; del buf118  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf145, (8, 1024, 64), (64, 512, 1), 0), permute_323, out=buf147)
    del permute_323
    buf148 = buf113; del buf113  # reuse
    buf149 = buf147; del buf147  # reuse
    buf150 = reinterpret_tensor(buf114, (8388608, ), (1, ), 0); del buf114  # reuse
    buf153 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_29(c_void_p(buf149.data_ptr()), c_void_p(getitem_95.data_ptr()), c_void_p(alias_79.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf153.data_ptr()))
    del alias_79
    del getitem_95
    buf155 = reinterpret_tensor(buf145, (8, 64, 1024), (65536, 1024, 1), 0); del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_324, buf153, out=buf155)
    del permute_324
    buf156 = reinterpret_tensor(buf127, (8, 1024, 64), (65536, 64, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf153, permute_325, out=buf156)
    del permute_325
    buf157 = reinterpret_tensor(buf124, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf124  # reuse
    cpp_fused_clone_30(c_void_p(tangents_18.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf157.data_ptr()))
    del tangents_18
    buf158 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (512, 1024), (1, 512), 0), view_169, out=buf158)
    buf159 = reinterpret_tensor(buf146, (1024, 512), (512, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (1024, 512), (512, 1), 0), permute_330, out=buf159)
    del permute_330
    buf160 = buf157; del buf157  # reuse
    cpp_fused_clone_31(c_void_p(tangents_17.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf160.data_ptr()))
    del tangents_17
    buf161 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf160, (512, 1024), (1, 512), 0), view_169, out=buf161)
    buf162 = reinterpret_tensor(buf155, (1024, 512), (512, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf160, (1024, 512), (512, 1), 0), permute_335, out=buf162)
    del permute_335
    buf163 = reinterpret_tensor(buf160, (1024, 512), (512, 1), 0); del buf160  # reuse
    cpp_fused_view_32(c_void_p(buf156.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (512, 1024), (1, 512), 0), view_298, out=buf164)
    del view_298
    buf165 = reinterpret_tensor(buf156, (1024, 512), (512, 1), 0); del buf156  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf163, permute_340, out=buf165)
    del permute_340
    buf166 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf167 = buf141; del buf141  # reuse
    buf168 = buf142; del buf142  # reuse
    buf169 = reinterpret_tensor(buf163, (1, 1024, 512), (524288, 512, 1), 0); del buf163  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_33(c_void_p(buf168.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(add_65.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(getitem_93.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()))
    del add_65
    del getitem_93
    del primals_24
    del rsqrt_23
    buf170 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (512, 1024), (1, 512), 0), view_296, out=buf170)
    del view_296
    buf171 = buf165; del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (1024, 512), (512, 1), 0), permute_344, out=buf171)
    del permute_344
    buf172 = reinterpret_tensor(buf169, (8, 1024, 64), (65536, 64, 1), 0); del buf169  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_347, reinterpret_tensor(buf171, (8, 1024, 64), (64, 512, 1), 0), out=buf172)
    del permute_347
    buf173 = buf153; del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf171, (8, 1024, 64), (64, 512, 1), 0), permute_348, out=buf173)
    del permute_348
    buf174 = buf148; del buf148  # reuse
    buf175 = buf173; del buf173  # reuse
    buf176 = buf150; del buf150  # reuse
    buf179 = buf149; del buf149  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_34(c_void_p(buf175.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(alias_81.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf179.data_ptr()))
    del alias_81
    del getitem_91
    buf181 = reinterpret_tensor(buf171, (8, 64, 1024), (65536, 1024, 1), 0); del buf171  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_349, buf179, out=buf181)
    del permute_349
    buf182 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf179, permute_350, out=buf182)
    del permute_350
    buf183 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_35(c_void_p(tangents_16.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf183.data_ptr()))
    del tangents_16
    buf184 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (512, 1024), (1, 512), 0), view_278, out=buf184)
    buf185 = reinterpret_tensor(buf172, (1024, 512), (512, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (1024, 512), (512, 1), 0), permute_355, out=buf185)
    del permute_355
    buf186 = buf183; del buf183  # reuse
    cpp_fused_clone_36(c_void_p(tangents_15.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf186.data_ptr()))
    del tangents_15
    buf187 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (512, 1024), (1, 512), 0), view_278, out=buf187)
    buf188 = reinterpret_tensor(buf181, (1024, 512), (512, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (1024, 512), (512, 1), 0), permute_360, out=buf188)
    del permute_360
    buf189 = reinterpret_tensor(buf186, (1024, 512), (512, 1), 0); del buf186  # reuse
    cpp_fused_view_37(c_void_p(buf182.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (512, 1024), (1, 512), 0), view_278, out=buf190)
    del view_278
    buf191 = reinterpret_tensor(buf182, (1024, 512), (512, 1), 0); del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf189, permute_365, out=buf191)
    del permute_365
    buf192 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf193 = buf167; del buf167  # reuse
    buf194 = buf168; del buf168  # reuse
    buf195 = reinterpret_tensor(buf189, (1, 1024, 512), (524288, 512, 1), 0); del buf189  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_38(c_void_p(buf194.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(add_62.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(getitem_89.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del add_62
    del getitem_89
    del primals_23
    del rsqrt_22
    buf196 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (512, 1024), (1, 512), 0), view_276, out=buf196)
    del view_276
    buf197 = reinterpret_tensor(buf137, (1024, 2048), (2048, 1), 0); del buf137  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (1024, 512), (512, 1), 0), permute_369, out=buf197)
    del permute_369
    buf198 = reinterpret_tensor(buf197, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf197  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_39(c_void_p(buf198.data_ptr()), c_void_p(le_4.data_ptr()), c_void_p(getitem_87.data_ptr()))
    del getitem_87
    del le_4
    buf199 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (2048, 1024), (1, 2048), 0), view_274, out=buf199)
    del view_274
    buf200 = reinterpret_tensor(buf195, (1024, 512), (512, 1), 0); del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (1024, 2048), (2048, 1), 0), permute_373, out=buf200)
    del permute_373
    buf201 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf202 = buf193; del buf193  # reuse
    buf203 = buf194; del buf194  # reuse
    buf204 = reinterpret_tensor(buf191, (1, 1024, 512), (524288, 512, 1), 0); del buf191  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_40(c_void_p(buf203.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(add_60.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(getitem_85.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()))
    del add_60
    del getitem_85
    del primals_22
    del rsqrt_21
    buf205 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (512, 1024), (1, 512), 0), view_272, out=buf205)
    del view_272
    buf206 = buf200; del buf200  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (1024, 512), (512, 1), 0), permute_377, out=buf206)
    del permute_377
    buf207 = reinterpret_tensor(buf204, (8, 1024, 64), (65536, 64, 1), 0); del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_380, reinterpret_tensor(buf206, (8, 1024, 64), (64, 512, 1), 0), out=buf207)
    del permute_380
    buf208 = buf179; del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf206, (8, 1024, 64), (64, 512, 1), 0), permute_381, out=buf208)
    del permute_381
    buf209 = buf174; del buf174  # reuse
    buf210 = buf208; del buf208  # reuse
    buf211 = reinterpret_tensor(buf175, (8388608, ), (1, ), 0); del buf175  # reuse
    buf214 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_41(c_void_p(buf210.data_ptr()), c_void_p(getitem_83.data_ptr()), c_void_p(alias_85.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf214.data_ptr()))
    del alias_85
    del getitem_83
    buf216 = reinterpret_tensor(buf206, (8, 64, 1024), (65536, 1024, 1), 0); del buf206  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_382, buf214, out=buf216)
    del permute_382
    buf217 = reinterpret_tensor(buf188, (8, 1024, 64), (65536, 64, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf214, permute_383, out=buf217)
    del permute_383
    buf218 = reinterpret_tensor(buf185, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf185  # reuse
    cpp_fused_clone_42(c_void_p(tangents_14.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf218.data_ptr()))
    del tangents_14
    buf219 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (512, 1024), (1, 512), 0), view_169, out=buf219)
    buf220 = reinterpret_tensor(buf207, (1024, 512), (512, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (1024, 512), (512, 1), 0), permute_388, out=buf220)
    del permute_388
    buf221 = buf218; del buf218  # reuse
    cpp_fused_clone_43(c_void_p(tangents_13.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf221.data_ptr()))
    del tangents_13
    buf222 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (512, 1024), (1, 512), 0), view_169, out=buf222)
    buf223 = reinterpret_tensor(buf216, (1024, 512), (512, 1), 0); del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (1024, 512), (512, 1), 0), permute_393, out=buf223)
    del permute_393
    buf225 = reinterpret_tensor(buf221, (1024, 512), (512, 1), 0); del buf221  # reuse
    cpp_fused_view_44(c_void_p(buf217.data_ptr()), c_void_p(buf225.data_ptr()))
    buf227 = reinterpret_tensor(buf217, (1024, 512), (512, 1), 0); del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf225, permute_398, out=buf227)
    del permute_398
    buf229 = buf202; del buf202  # reuse
    buf230 = buf203; del buf203  # reuse
    buf231 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_45(c_void_p(buf230.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(add_57.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del getitem_81
    del primals_21
    buf233 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (1024, 512), (512, 1), 0), permute_402, out=buf233)
    del permute_402
    buf234 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_405, reinterpret_tensor(buf233, (8, 1024, 64), (64, 512, 1), 0), out=buf234)
    del permute_405
    buf245 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_46(c_void_p(tangents_12.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf245.data_ptr()))
    del tangents_12
    buf247 = reinterpret_tensor(buf234, (1024, 512), (512, 1), 0); del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (1024, 512), (512, 1), 0), permute_413, out=buf247)
    del permute_413
    buf235 = buf214; del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (8, 1024, 64), (64, 512, 1), 0), permute_406, out=buf235)
    del permute_406
    buf236 = buf209; del buf209  # reuse
    buf237 = buf235; del buf235  # reuse
    buf238 = buf211; del buf211  # reuse
    buf241 = buf210; del buf210  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_47(c_void_p(buf237.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(alias_87.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf241.data_ptr()))
    del alias_87
    del getitem_79
    buf243 = reinterpret_tensor(buf233, (8, 64, 1024), (65536, 1024, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_407, buf241, out=buf243)
    del permute_407
    buf248 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_48(c_void_p(tangents_11.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf248.data_ptr()))
    del tangents_11
    buf250 = reinterpret_tensor(buf243, (1024, 512), (512, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (1024, 512), (512, 1), 0), permute_418, out=buf250)
    del permute_418
    buf244 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf241, permute_408, out=buf244)
    del permute_408
    buf251 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_49(c_void_p(buf244.data_ptr()), c_void_p(buf251.data_ptr()))
    buf253 = reinterpret_tensor(buf244, (1024, 512), (512, 1), 0); del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf251, permute_423, out=buf253)
    del permute_423
    buf255 = buf229; del buf229  # reuse
    buf256 = buf230; del buf230  # reuse
    buf257 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_50(c_void_p(buf256.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(add_54.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del getitem_77
    del primals_20
    buf259 = reinterpret_tensor(buf198, (1024, 2048), (2048, 1), 0); del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1024, 512), (512, 1), 0), permute_427, out=buf259)
    del permute_427
    buf260 = reinterpret_tensor(buf259, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf259  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_51(c_void_p(buf260.data_ptr()), c_void_p(le_5.data_ptr()), c_void_p(getitem_75.data_ptr()))
    del getitem_75
    del le_5
    buf262 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (1024, 2048), (2048, 1), 0), permute_431, out=buf262)
    del permute_431
    buf264 = buf255; del buf255  # reuse
    buf265 = buf256; del buf256  # reuse
    buf266 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_52(c_void_p(buf265.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(add_52.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del getitem_73
    del primals_19
    buf268 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1024, 512), (512, 1), 0), permute_435, out=buf268)
    del permute_435
    buf269 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_438, reinterpret_tensor(buf268, (8, 1024, 64), (64, 512, 1), 0), out=buf269)
    del permute_438
    buf280 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_53(c_void_p(tangents_10.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf280.data_ptr()))
    del tangents_10
    buf282 = reinterpret_tensor(buf269, (1024, 512), (512, 1), 0); del buf269  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (1024, 512), (512, 1), 0), permute_446, out=buf282)
    del permute_446
    buf270 = buf241; del buf241  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (8, 1024, 64), (64, 512, 1), 0), permute_439, out=buf270)
    del permute_439
    buf271 = buf236; del buf236  # reuse
    buf272 = buf270; del buf270  # reuse
    buf273 = reinterpret_tensor(buf237, (8388608, ), (1, ), 0); del buf237  # reuse
    buf276 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_54(c_void_p(buf272.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(alias_91.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf276.data_ptr()))
    del alias_91
    del getitem_71
    buf278 = reinterpret_tensor(buf268, (8, 64, 1024), (65536, 1024, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_440, buf276, out=buf278)
    del permute_440
    buf283 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_55(c_void_p(tangents_9.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf283.data_ptr()))
    del tangents_9
    buf285 = reinterpret_tensor(buf278, (1024, 512), (512, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (1024, 512), (512, 1), 0), permute_451, out=buf285)
    del permute_451
    buf279 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf276, permute_441, out=buf279)
    del permute_441
    buf286 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_56(c_void_p(buf279.data_ptr()), c_void_p(buf286.data_ptr()))
    buf288 = reinterpret_tensor(buf279, (1024, 512), (512, 1), 0); del buf279  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf286, permute_456, out=buf288)
    del permute_456
    buf290 = buf264; del buf264  # reuse
    buf291 = buf265; del buf265  # reuse
    buf292 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_57(c_void_p(buf291.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(add_49.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()))
    del getitem_69
    del primals_18
    buf294 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (1024, 512), (512, 1), 0), permute_460, out=buf294)
    del permute_460
    buf295 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_463, reinterpret_tensor(buf294, (8, 1024, 64), (64, 512, 1), 0), out=buf295)
    del permute_463
    buf306 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_58(c_void_p(tangents_8.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf306.data_ptr()))
    del tangents_8
    buf308 = reinterpret_tensor(buf295, (1024, 512), (512, 1), 0); del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (1024, 512), (512, 1), 0), permute_471, out=buf308)
    del permute_471
    buf296 = buf276; del buf276  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf294, (8, 1024, 64), (64, 512, 1), 0), permute_464, out=buf296)
    del permute_464
    buf297 = buf271; del buf271  # reuse
    buf298 = buf296; del buf296  # reuse
    buf299 = buf273; del buf273  # reuse
    buf302 = buf272; del buf272  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_59(c_void_p(buf298.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(alias_93.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf302.data_ptr()))
    del alias_93
    del getitem_67
    buf304 = reinterpret_tensor(buf294, (8, 64, 1024), (65536, 1024, 1), 0); del buf294  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_465, buf302, out=buf304)
    del permute_465
    buf309 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_60(c_void_p(tangents_7.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf309.data_ptr()))
    del tangents_7
    buf311 = reinterpret_tensor(buf304, (1024, 512), (512, 1), 0); del buf304  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (1024, 512), (512, 1), 0), permute_476, out=buf311)
    del permute_476
    buf305 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf302, permute_466, out=buf305)
    del permute_466
    buf312 = empty((1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_61(c_void_p(buf305.data_ptr()), c_void_p(buf312.data_ptr()))
    buf314 = reinterpret_tensor(buf305, (1024, 512), (512, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf312, permute_481, out=buf314)
    del permute_481
    buf316 = buf290; del buf290  # reuse
    buf317 = buf291; del buf291  # reuse
    buf318 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_62(c_void_p(buf317.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(add_46.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(getitem_65.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del getitem_65
    del primals_17
    buf320 = empty((1024, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (1024, 512), (512, 1), 0), permute_485, out=buf320)
    del permute_485
    buf321 = reinterpret_tensor(buf320, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf320  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_63(c_void_p(buf321.data_ptr()), c_void_p(le_6.data_ptr()), c_void_p(getitem_63.data_ptr()))
    del getitem_63
    del le_6
    buf323 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (1024, 2048), (2048, 1), 0), permute_489, out=buf323)
    del permute_489
    buf325 = buf316; del buf316  # reuse
    buf326 = buf317; del buf317  # reuse
    buf327 = empty((1, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_64(c_void_p(buf326.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(add_44.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()))
    del getitem_61
    del primals_16
    buf329 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (1024, 512), (512, 1), 0), permute_493, out=buf329)
    del permute_493
    buf330 = empty((8, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_496, reinterpret_tensor(buf329, (8, 1024, 64), (64, 512, 1), 0), out=buf330)
    del permute_496
    buf341 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_65(c_void_p(tangents_6.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf341.data_ptr()))
    del tangents_6
    buf343 = reinterpret_tensor(buf330, (1024, 512), (512, 1), 0); del buf330  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (1024, 512), (512, 1), 0), permute_504, out=buf343)
    del permute_504
    buf331 = buf302; del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (8, 1024, 64), (64, 512, 1), 0), permute_497, out=buf331)
    del permute_497
    buf332 = buf297; del buf297  # reuse
    buf333 = buf331; del buf331  # reuse
    buf334 = reinterpret_tensor(buf298, (8388608, ), (1, ), 0); del buf298  # reuse
    buf337 = empty((8, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_66(c_void_p(buf333.data_ptr()), c_void_p(getitem_59.data_ptr()), c_void_p(alias_97.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf337.data_ptr()))
    del alias_97
    del getitem_59
    buf339 = reinterpret_tensor(buf329, (8, 64, 1024), (65536, 1024, 1), 0); del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_498, buf337, out=buf339)
    del permute_498
    buf344 = empty((1, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_67(c_void_p(tangents_5.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf344.data_ptr()))
    del tangents_5
    buf346 = reinterpret_tensor(buf339, (1024, 512), (512, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (1024, 512), (512, 1), 0), permute_509, out=buf346)
    del permute_509
    buf224 = reinterpret_tensor(buf101, (1, 1024, 512), (524288, 512, 1), 0); del buf101  # reuse
    buf387 = buf224; del buf224  # reuse
    cpp_fused_add_native_dropout_backward_68(c_void_p(buf387.data_ptr()), c_void_p(tangents_27.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(getitem_51.data_ptr()))
    del buf159
    del buf162
    del buf220
    del buf223
    del buf282
    del buf285
    del buf343
    del buf346
    del buf37
    del buf40
    del buf98
    del getitem_51
    del tangents_27
    buf226 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf225, (512, 1024), (1, 512), 0), view_254, out=buf226)
    del buf225
    del view_254
    buf228 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_69(c_void_p(buf227.data_ptr()), c_void_p(add_57.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf228.data_ptr()))
    del add_57
    del buf227
    del rsqrt_20
    buf232 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (512, 1024), (1, 512), 0), view_252, out=buf232)
    del buf231
    del view_252
    buf246 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (512, 1024), (1, 512), 0), view_234, out=buf246)
    del buf245
    buf249 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (512, 1024), (1, 512), 0), view_234, out=buf249)
    del buf248
    buf252 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (512, 1024), (1, 512), 0), view_234, out=buf252)
    del buf251
    del view_234
    buf254 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_70(c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(add_54.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf254.data_ptr()))
    del add_54
    del buf247
    del buf250
    del buf253
    del rsqrt_19
    buf258 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (512, 1024), (1, 512), 0), view_232, out=buf258)
    del buf257
    del view_232
    buf261 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (2048, 1024), (1, 2048), 0), view_230, out=buf261)
    del buf260
    del view_230
    buf263 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_71(c_void_p(buf262.data_ptr()), c_void_p(add_52.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf263.data_ptr()))
    del add_52
    del buf262
    del rsqrt_18
    buf267 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (512, 1024), (1, 512), 0), view_228, out=buf267)
    del buf266
    del view_228
    buf281 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (512, 1024), (1, 512), 0), view_169, out=buf281)
    del buf280
    buf284 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (512, 1024), (1, 512), 0), view_169, out=buf284)
    del buf283
    buf287 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (512, 1024), (1, 512), 0), view_210, out=buf287)
    del buf286
    del view_210
    buf289 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_72(c_void_p(buf288.data_ptr()), c_void_p(add_49.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(buf289.data_ptr()))
    del add_49
    del buf288
    del rsqrt_17
    buf293 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (512, 1024), (1, 512), 0), view_208, out=buf293)
    del buf292
    del view_208
    buf307 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (512, 1024), (1, 512), 0), view_190, out=buf307)
    del buf306
    buf310 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (512, 1024), (1, 512), 0), view_190, out=buf310)
    del buf309
    buf313 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (512, 1024), (1, 512), 0), view_190, out=buf313)
    del buf312
    del view_190
    buf315 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_73(c_void_p(buf308.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(add_46.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf315.data_ptr()))
    del add_46
    del buf308
    del buf311
    del buf314
    del rsqrt_16
    buf319 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (512, 1024), (1, 512), 0), view_188, out=buf319)
    del buf318
    del view_188
    buf322 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (2048, 1024), (1, 2048), 0), view_186, out=buf322)
    del view_186
    buf324 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_74(c_void_p(buf323.data_ptr()), c_void_p(add_44.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(buf324.data_ptr()))
    del add_44
    del rsqrt_15
    buf328 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (512, 1024), (1, 512), 0), view_184, out=buf328)
    del view_184
    buf340 = reinterpret_tensor(buf327, (8, 1024, 64), (65536, 64, 1), 0); del buf327  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf337, permute_499, out=buf340)
    del permute_499
    buf342 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (512, 1024), (1, 512), 0), view_169, out=buf342)
    buf345 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (512, 1024), (1, 512), 0), view_169, out=buf345)
    del view_169
    buf347 = reinterpret_tensor(buf344, (1024, 512), (512, 1), 0); del buf344  # reuse
    cpp_fused_view_75(c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf347, (512, 1024), (1, 512), 0), view_166, out=buf348)
    del view_166
    buf349 = reinterpret_tensor(buf340, (1024, 512), (512, 1), 0); del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf347, permute_514, out=buf349)
    del permute_514
    buf350 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf351 = buf325; del buf325  # reuse
    buf352 = buf326; del buf326  # reuse
    buf353 = reinterpret_tensor(buf347, (1, 1024, 512), (524288, 512, 1), 0); del buf347  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_76(c_void_p(buf352.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(add_40.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()))
    del add_40
    del getitem_57
    del primals_15
    del rsqrt_14
    buf354 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (512, 1024), (1, 512), 0), view_164, out=buf354)
    del view_164
    buf355 = buf349; del buf349  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (1024, 512), (512, 1), 0), permute_518, out=buf355)
    del permute_518
    buf356 = reinterpret_tensor(buf353, (8, 1024, 64), (65536, 64, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_521, reinterpret_tensor(buf355, (8, 1024, 64), (64, 512, 1), 0), out=buf356)
    del permute_521
    buf357 = buf337; del buf337  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf355, (8, 1024, 64), (64, 512, 1), 0), permute_522, out=buf357)
    del permute_522
    buf358 = buf332; del buf332  # reuse
    buf359 = buf357; del buf357  # reuse
    buf360 = buf334; del buf334  # reuse
    buf363 = buf333; del buf333  # reuse
    buf366 = empty_strided((1024, 1024, 8), (1024, 1, 1048576), device='cpu', dtype=torch.float32)
    buf365 = empty((32, 8), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_77(c_void_p(buf359.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(alias_99.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf365.data_ptr()))
    del alias_99
    del getitem_55
    aten.index_put_(buf365, [add_37], buf366, True)
    del add_37
    buf369 = reinterpret_tensor(buf355, (8, 64, 1024), (65536, 1024, 1), 0); del buf355  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_524, buf363, out=buf369)
    del permute_524
    buf370 = reinterpret_tensor(buf341, (8, 1024, 64), (65536, 64, 1), 0); del buf341  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf363, permute_525, out=buf370)
    del permute_525
    buf371 = reinterpret_tensor(buf323, (1, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf323  # reuse
    cpp_fused_clone_78(c_void_p(tangents_4.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf371.data_ptr()))
    del tangents_4
    buf372 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (512, 1024), (1, 512), 0), view_146, out=buf372)
    buf373 = reinterpret_tensor(buf356, (1024, 512), (512, 1), 0); del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (1024, 512), (512, 1), 0), permute_530, out=buf373)
    del permute_530
    buf374 = buf371; del buf371  # reuse
    cpp_fused_clone_79(c_void_p(tangents_3.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf374.data_ptr()))
    del tangents_3
    buf375 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (512, 1024), (1, 512), 0), view_146, out=buf375)
    buf376 = reinterpret_tensor(buf369, (1024, 512), (512, 1), 0); del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (1024, 512), (512, 1), 0), permute_535, out=buf376)
    del permute_535
    buf377 = reinterpret_tensor(buf374, (1024, 512), (512, 1), 0); del buf374  # reuse
    cpp_fused_view_80(c_void_p(buf370.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (512, 1024), (1, 512), 0), view_146, out=buf378)
    del view_146
    buf379 = reinterpret_tensor(buf370, (1024, 512), (512, 1), 0); del buf370  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf377, permute_540, out=buf379)
    del buf377
    del permute_540
    buf380 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf381 = buf351; del buf351  # reuse
    buf382 = buf352; del buf352  # reuse
    buf384 = buf382; del buf382  # reuse
    buf383 = empty((32128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_81(c_void_p(buf384.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(getitem_52.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(view_145.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()))
    del getitem_52
    del getitem_53
    del primals_14
    del rsqrt_13
    aten.index_put_(buf383, [view_145], buf384, True)
    del view_145
    buf388 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf389 = buf381; del buf381  # reuse
    buf390 = buf387; del buf387  # reuse
    buf391 = buf384; del buf384  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_82(c_void_p(buf390.data_ptr()), c_void_p(add_33.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()))
    del add_33
    del getitem_49
    del primals_13
    del rsqrt_12
    buf392 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (512, 1024), (1, 512), 0), view_143, out=buf392)
    del view_143
    buf393 = reinterpret_tensor(buf321, (1024, 2048), (2048, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (1024, 512), (512, 1), 0), permute_544, out=buf393)
    del permute_544
    buf394 = reinterpret_tensor(buf393, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf393  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_83(c_void_p(buf394.data_ptr()), c_void_p(le_7.data_ptr()), c_void_p(getitem_47.data_ptr()))
    del getitem_47
    del le_7
    buf395 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (2048, 1024), (1, 2048), 0), view_141, out=buf395)
    del view_141
    buf396 = reinterpret_tensor(buf391, (1024, 512), (512, 1), 0); del buf391  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (1024, 2048), (2048, 1), 0), permute_548, out=buf396)
    del permute_548
    buf397 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf398 = buf389; del buf389  # reuse
    buf399 = buf390; del buf390  # reuse
    buf400 = reinterpret_tensor(buf379, (1, 1024, 512), (524288, 512, 1), 0); del buf379  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_84(c_void_p(buf399.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(add_31.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf400.data_ptr()))
    del add_31
    del getitem_45
    del primals_12
    del rsqrt_11
    buf401 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf400, (512, 1024), (1, 512), 0), view_139, out=buf401)
    del view_139
    buf402 = buf396; del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf400, (1024, 512), (512, 1), 0), permute_552, out=buf402)
    del permute_552
    buf403 = reinterpret_tensor(buf400, (8, 1024, 64), (65536, 64, 1), 0); del buf400  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_555, reinterpret_tensor(buf402, (8, 1024, 64), (64, 512, 1), 0), out=buf403)
    del permute_555
    buf404 = buf363; del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf402, (8, 1024, 64), (64, 512, 1), 0), permute_556, out=buf404)
    del permute_556
    buf405 = buf358; del buf358  # reuse
    buf406 = buf404; del buf404  # reuse
    buf407 = reinterpret_tensor(buf366, (8388608, ), (1, ), 0); del buf366  # reuse
    buf410 = reinterpret_tensor(buf54, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf54  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_85(c_void_p(buf406.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(alias_104.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf410.data_ptr()))
    del alias_104
    del getitem_43
    buf412 = reinterpret_tensor(buf402, (8, 64, 1024), (65536, 1024, 1), 0); del buf402  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_557, buf410, out=buf412)
    del permute_557
    buf413 = reinterpret_tensor(buf376, (8, 1024, 64), (65536, 64, 1), 0); del buf376  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf410, permute_558, out=buf413)
    del permute_558
    buf414 = buf373; del buf373  # reuse
    cpp_fused_view_86(c_void_p(buf403.data_ptr()), c_void_p(buf414.data_ptr()))
    buf415 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf414, (512, 1024), (1, 512), 0), view_121, out=buf415)
    buf416 = reinterpret_tensor(buf403, (1024, 512), (512, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf414, permute_563, out=buf416)
    del permute_563
    buf417 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (512, 1024), (1024, 1), 0), view_121, out=buf417)
    buf418 = buf414; del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (1024, 512), (1, 1024), 0), permute_568, out=buf418)
    del permute_568
    buf419 = reinterpret_tensor(buf412, (1024, 512), (512, 1), 0); del buf412  # reuse
    cpp_fused_view_87(c_void_p(buf413.data_ptr()), c_void_p(buf419.data_ptr()))
    buf420 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (512, 1024), (1, 512), 0), view_121, out=buf420)
    del view_121
    buf421 = reinterpret_tensor(buf413, (1024, 512), (512, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf419, permute_573, out=buf421)
    del permute_573
    buf422 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf423 = buf398; del buf398  # reuse
    buf424 = buf399; del buf399  # reuse
    buf425 = reinterpret_tensor(buf419, (1, 1024, 512), (524288, 512, 1), 0); del buf419  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_88(c_void_p(buf424.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(add_28.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf425.data_ptr()))
    del add_28
    del getitem_41
    del primals_11
    del rsqrt_10
    buf426 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (512, 1024), (1, 512), 0), view_119, out=buf426)
    del view_119
    buf427 = reinterpret_tensor(buf394, (1024, 2048), (2048, 1), 0); del buf394  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (1024, 512), (512, 1), 0), permute_577, out=buf427)
    del permute_577
    buf428 = reinterpret_tensor(buf427, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf427  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_89(c_void_p(buf428.data_ptr()), c_void_p(le_8.data_ptr()), c_void_p(getitem_39.data_ptr()))
    del getitem_39
    del le_8
    buf429 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf428, (2048, 1024), (1, 2048), 0), view_117, out=buf429)
    del view_117
    buf430 = reinterpret_tensor(buf425, (1024, 512), (512, 1), 0); del buf425  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf428, (1024, 2048), (2048, 1), 0), permute_581, out=buf430)
    del permute_581
    buf431 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf432 = buf423; del buf423  # reuse
    buf433 = buf424; del buf424  # reuse
    buf434 = reinterpret_tensor(buf421, (1, 1024, 512), (524288, 512, 1), 0); del buf421  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_90(c_void_p(buf433.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(add_26.data_ptr()), c_void_p(rsqrt_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()))
    del add_26
    del getitem_37
    del primals_10
    del rsqrt_9
    buf435 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf434, (512, 1024), (1, 512), 0), view_115, out=buf435)
    del view_115
    buf436 = buf430; del buf430  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf434, (1024, 512), (512, 1), 0), permute_585, out=buf436)
    del permute_585
    buf437 = reinterpret_tensor(buf434, (8, 1024, 64), (65536, 64, 1), 0); del buf434  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_588, reinterpret_tensor(buf436, (8, 1024, 64), (64, 512, 1), 0), out=buf437)
    del permute_588
    buf438 = buf410; del buf410  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf436, (8, 1024, 64), (64, 512, 1), 0), permute_589, out=buf438)
    del permute_589
    buf439 = buf405; del buf405  # reuse
    buf440 = buf438; del buf438  # reuse
    buf441 = reinterpret_tensor(buf406, (8388608, ), (1, ), 0); del buf406  # reuse
    buf444 = reinterpret_tensor(buf360, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf360  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_91(c_void_p(buf440.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(alias_108.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf444.data_ptr()))
    del alias_108
    del getitem_35
    buf446 = reinterpret_tensor(buf436, (8, 64, 1024), (65536, 1024, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_590, buf444, out=buf446)
    del permute_590
    buf447 = reinterpret_tensor(buf418, (8, 1024, 64), (65536, 64, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf444, permute_591, out=buf447)
    del permute_591
    buf448 = buf416; del buf416  # reuse
    cpp_fused_view_92(c_void_p(buf437.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (512, 1024), (1, 512), 0), view_97, out=buf449)
    buf450 = reinterpret_tensor(buf437, (1024, 512), (512, 1), 0); del buf437  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf448, permute_596, out=buf450)
    del permute_596
    buf451 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (512, 1024), (1024, 1), 0), view_97, out=buf451)
    buf452 = buf448; del buf448  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (1024, 512), (1, 1024), 0), permute_601, out=buf452)
    del permute_601
    buf453 = reinterpret_tensor(buf446, (1024, 512), (512, 1), 0); del buf446  # reuse
    cpp_fused_view_93(c_void_p(buf447.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (512, 1024), (1, 512), 0), view_97, out=buf454)
    del view_97
    buf455 = reinterpret_tensor(buf447, (1024, 512), (512, 1), 0); del buf447  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf453, permute_606, out=buf455)
    del permute_606
    buf456 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf457 = buf432; del buf432  # reuse
    buf458 = buf433; del buf433  # reuse
    buf459 = reinterpret_tensor(buf453, (1, 1024, 512), (524288, 512, 1), 0); del buf453  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_94(c_void_p(buf458.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(add_23.data_ptr()), c_void_p(rsqrt_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf459.data_ptr()))
    del add_23
    del getitem_33
    del primals_9
    del rsqrt_8
    buf460 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (512, 1024), (1, 512), 0), view_95, out=buf460)
    del view_95
    buf461 = reinterpret_tensor(buf428, (1024, 2048), (2048, 1), 0); del buf428  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (1024, 512), (512, 1), 0), permute_610, out=buf461)
    del permute_610
    buf462 = reinterpret_tensor(buf461, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf461  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_95(c_void_p(buf462.data_ptr()), c_void_p(le_9.data_ptr()), c_void_p(getitem_31.data_ptr()))
    del getitem_31
    del le_9
    buf463 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (2048, 1024), (1, 2048), 0), view_93, out=buf463)
    del view_93
    buf464 = reinterpret_tensor(buf459, (1024, 512), (512, 1), 0); del buf459  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (1024, 2048), (2048, 1), 0), permute_614, out=buf464)
    del permute_614
    buf465 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf466 = buf457; del buf457  # reuse
    buf467 = buf458; del buf458  # reuse
    buf468 = reinterpret_tensor(buf455, (1, 1024, 512), (524288, 512, 1), 0); del buf455  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_96(c_void_p(buf467.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(add_21.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()))
    del add_21
    del getitem_29
    del primals_8
    del rsqrt_7
    buf469 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (512, 1024), (1, 512), 0), view_91, out=buf469)
    del view_91
    buf470 = buf464; del buf464  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (1024, 512), (512, 1), 0), permute_618, out=buf470)
    del permute_618
    buf471 = reinterpret_tensor(buf468, (8, 1024, 64), (65536, 64, 1), 0); del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_621, reinterpret_tensor(buf470, (8, 1024, 64), (64, 512, 1), 0), out=buf471)
    del permute_621
    buf472 = buf444; del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf470, (8, 1024, 64), (64, 512, 1), 0), permute_622, out=buf472)
    del permute_622
    buf473 = buf439; del buf439  # reuse
    buf474 = buf472; del buf472  # reuse
    buf475 = reinterpret_tensor(buf440, (8388608, ), (1, ), 0); del buf440  # reuse
    buf478 = buf359; del buf359  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_97(c_void_p(buf474.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(alias_112.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf478.data_ptr()))
    del alias_112
    del getitem_27
    buf480 = reinterpret_tensor(buf470, (8, 64, 1024), (65536, 1024, 1), 0); del buf470  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_623, buf478, out=buf480)
    del permute_623
    buf481 = reinterpret_tensor(buf452, (8, 1024, 64), (65536, 64, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf478, permute_624, out=buf481)
    del permute_624
    buf482 = buf450; del buf450  # reuse
    cpp_fused_view_98(c_void_p(buf471.data_ptr()), c_void_p(buf482.data_ptr()))
    buf483 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (512, 1024), (1, 512), 0), view_73, out=buf483)
    buf484 = reinterpret_tensor(buf471, (1024, 512), (512, 1), 0); del buf471  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf482, permute_629, out=buf484)
    del permute_629
    buf485 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (512, 1024), (1024, 1), 0), view_73, out=buf485)
    buf486 = buf482; del buf482  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (1024, 512), (1, 1024), 0), permute_634, out=buf486)
    del permute_634
    buf487 = reinterpret_tensor(buf480, (1024, 512), (512, 1), 0); del buf480  # reuse
    cpp_fused_view_99(c_void_p(buf481.data_ptr()), c_void_p(buf487.data_ptr()))
    buf488 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf487, (512, 1024), (1, 512), 0), view_73, out=buf488)
    del view_73
    buf489 = reinterpret_tensor(buf481, (1024, 512), (512, 1), 0); del buf481  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf487, permute_639, out=buf489)
    del permute_639
    buf490 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf491 = buf466; del buf466  # reuse
    buf492 = buf467; del buf467  # reuse
    buf493 = reinterpret_tensor(buf487, (1, 1024, 512), (524288, 512, 1), 0); del buf487  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_100(c_void_p(buf492.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(add_18.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()))
    del add_18
    del getitem_25
    del primals_7
    del rsqrt_6
    buf494 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (512, 1024), (1, 512), 0), view_71, out=buf494)
    del view_71
    buf495 = reinterpret_tensor(buf462, (1024, 2048), (2048, 1), 0); del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (1024, 512), (512, 1), 0), permute_643, out=buf495)
    del permute_643
    buf496 = reinterpret_tensor(buf495, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf495  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_101(c_void_p(buf496.data_ptr()), c_void_p(le_10.data_ptr()), c_void_p(getitem_23.data_ptr()))
    del getitem_23
    del le_10
    buf497 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf496, (2048, 1024), (1, 2048), 0), view_69, out=buf497)
    del view_69
    buf498 = reinterpret_tensor(buf493, (1024, 512), (512, 1), 0); del buf493  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf496, (1024, 2048), (2048, 1), 0), permute_647, out=buf498)
    del permute_647
    buf499 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf500 = buf491; del buf491  # reuse
    buf501 = buf492; del buf492  # reuse
    buf502 = reinterpret_tensor(buf489, (1, 1024, 512), (524288, 512, 1), 0); del buf489  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_102(c_void_p(buf501.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(add_16.data_ptr()), c_void_p(rsqrt_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf502.data_ptr()))
    del add_16
    del getitem_21
    del primals_6
    del rsqrt_5
    buf503 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf502, (512, 1024), (1, 512), 0), view_67, out=buf503)
    del view_67
    buf504 = buf498; del buf498  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf502, (1024, 512), (512, 1), 0), permute_651, out=buf504)
    del permute_651
    buf505 = reinterpret_tensor(buf502, (8, 1024, 64), (65536, 64, 1), 0); del buf502  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_654, reinterpret_tensor(buf504, (8, 1024, 64), (64, 512, 1), 0), out=buf505)
    del permute_654
    buf506 = buf478; del buf478  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf504, (8, 1024, 64), (64, 512, 1), 0), permute_655, out=buf506)
    del permute_655
    buf507 = buf473; del buf473  # reuse
    buf508 = buf506; del buf506  # reuse
    buf509 = reinterpret_tensor(buf474, (8388608, ), (1, ), 0); del buf474  # reuse
    buf512 = reinterpret_tensor(buf299, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf299  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_103(c_void_p(buf508.data_ptr()), c_void_p(getitem_19.data_ptr()), c_void_p(alias_116.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf512.data_ptr()))
    del alias_116
    del getitem_19
    buf514 = reinterpret_tensor(buf504, (8, 64, 1024), (65536, 1024, 1), 0); del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_656, buf512, out=buf514)
    del permute_656
    buf515 = reinterpret_tensor(buf486, (8, 1024, 64), (65536, 64, 1), 0); del buf486  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf512, permute_657, out=buf515)
    del permute_657
    buf516 = buf484; del buf484  # reuse
    cpp_fused_view_104(c_void_p(buf505.data_ptr()), c_void_p(buf516.data_ptr()))
    buf517 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf516, (512, 1024), (1, 512), 0), view_49, out=buf517)
    buf518 = reinterpret_tensor(buf505, (1024, 512), (512, 1), 0); del buf505  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf516, permute_662, out=buf518)
    del permute_662
    buf519 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0), view_49, out=buf519)
    buf520 = buf516; del buf516  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf514, (1024, 512), (1, 1024), 0), permute_667, out=buf520)
    del permute_667
    buf521 = reinterpret_tensor(buf514, (1024, 512), (512, 1), 0); del buf514  # reuse
    cpp_fused_view_105(c_void_p(buf515.data_ptr()), c_void_p(buf521.data_ptr()))
    buf522 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (512, 1024), (1, 512), 0), view_49, out=buf522)
    del view_49
    buf523 = reinterpret_tensor(buf515, (1024, 512), (512, 1), 0); del buf515  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf521, permute_672, out=buf523)
    del permute_672
    buf524 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf525 = buf500; del buf500  # reuse
    buf526 = buf501; del buf501  # reuse
    buf527 = reinterpret_tensor(buf521, (1, 1024, 512), (524288, 512, 1), 0); del buf521  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_106(c_void_p(buf526.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(add_13.data_ptr()), c_void_p(rsqrt_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf527.data_ptr()))
    del add_13
    del getitem_17
    del primals_5
    del rsqrt_4
    buf528 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (512, 1024), (1, 512), 0), view_47, out=buf528)
    del view_47
    buf529 = reinterpret_tensor(buf496, (1024, 2048), (2048, 1), 0); del buf496  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (1024, 512), (512, 1), 0), permute_676, out=buf529)
    del permute_676
    buf530 = reinterpret_tensor(buf529, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf529  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_107(c_void_p(buf530.data_ptr()), c_void_p(le_11.data_ptr()), c_void_p(getitem_15.data_ptr()))
    del getitem_15
    del le_11
    buf531 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf530, (2048, 1024), (1, 2048), 0), view_45, out=buf531)
    del view_45
    buf532 = reinterpret_tensor(buf527, (1024, 512), (512, 1), 0); del buf527  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf530, (1024, 2048), (2048, 1), 0), permute_680, out=buf532)
    del permute_680
    buf533 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf534 = buf525; del buf525  # reuse
    buf535 = buf526; del buf526  # reuse
    buf536 = reinterpret_tensor(buf523, (1, 1024, 512), (524288, 512, 1), 0); del buf523  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_108(c_void_p(buf535.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(add_11.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf536.data_ptr()))
    del add_11
    del getitem_13
    del primals_4
    del rsqrt_3
    buf537 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (512, 1024), (1, 512), 0), view_43, out=buf537)
    del view_43
    buf538 = buf532; del buf532  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (1024, 512), (512, 1), 0), permute_684, out=buf538)
    del permute_684
    buf539 = reinterpret_tensor(buf536, (8, 1024, 64), (65536, 64, 1), 0); del buf536  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_687, reinterpret_tensor(buf538, (8, 1024, 64), (64, 512, 1), 0), out=buf539)
    del permute_687
    buf540 = buf512; del buf512  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf538, (8, 1024, 64), (64, 512, 1), 0), permute_688, out=buf540)
    del permute_688
    buf541 = buf507; del buf507  # reuse
    buf542 = buf540; del buf540  # reuse
    buf543 = reinterpret_tensor(buf508, (8388608, ), (1, ), 0); del buf508  # reuse
    buf546 = reinterpret_tensor(buf238, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf238  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_109(c_void_p(buf542.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(alias_120.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf546.data_ptr()))
    del alias_120
    del getitem_11
    buf548 = reinterpret_tensor(buf538, (8, 64, 1024), (65536, 1024, 1), 0); del buf538  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_689, buf546, out=buf548)
    del permute_689
    buf549 = reinterpret_tensor(buf520, (8, 1024, 64), (65536, 64, 1), 0); del buf520  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf546, permute_690, out=buf549)
    del permute_690
    buf550 = buf518; del buf518  # reuse
    cpp_fused_view_110(c_void_p(buf539.data_ptr()), c_void_p(buf550.data_ptr()))
    buf551 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (512, 1024), (1, 512), 0), view_25, out=buf551)
    buf552 = reinterpret_tensor(buf539, (1024, 512), (512, 1), 0); del buf539  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf550, permute_695, out=buf552)
    del permute_695
    buf553 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf548, (512, 1024), (1024, 1), 0), view_25, out=buf553)
    buf554 = buf550; del buf550  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf548, (1024, 512), (1, 1024), 0), permute_700, out=buf554)
    del permute_700
    buf555 = reinterpret_tensor(buf548, (1024, 512), (512, 1), 0); del buf548  # reuse
    cpp_fused_view_111(c_void_p(buf549.data_ptr()), c_void_p(buf555.data_ptr()))
    buf556 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf555, (512, 1024), (1, 512), 0), view_25, out=buf556)
    del view_25
    buf557 = reinterpret_tensor(buf549, (1024, 512), (512, 1), 0); del buf549  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf555, permute_705, out=buf557)
    del permute_705
    buf558 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf559 = buf534; del buf534  # reuse
    buf560 = buf535; del buf535  # reuse
    buf561 = reinterpret_tensor(buf555, (1, 1024, 512), (524288, 512, 1), 0); del buf555  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_112(c_void_p(buf560.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(add_8.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(getitem_9.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf561.data_ptr()))
    del add_8
    del getitem_9
    del primals_3
    del rsqrt_2
    buf562 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (512, 1024), (1, 512), 0), view_23, out=buf562)
    del view_23
    buf563 = reinterpret_tensor(buf530, (1024, 2048), (2048, 1), 0); del buf530  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (1024, 512), (512, 1), 0), permute_709, out=buf563)
    del permute_709
    buf564 = reinterpret_tensor(buf563, (1, 1024, 2048), (2097152, 2048, 1), 0); del buf563  # reuse
    cpp_fused_native_dropout_backward_nll_loss_forward_threshold_backward_113(c_void_p(buf564.data_ptr()), c_void_p(le_12.data_ptr()), c_void_p(getitem_7.data_ptr()))
    del getitem_7
    del le_12
    buf565 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf564, (2048, 1024), (1, 2048), 0), view_21, out=buf565)
    del view_21
    buf566 = reinterpret_tensor(buf561, (1024, 512), (512, 1), 0); del buf561  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf564, (1024, 2048), (2048, 1), 0), permute_713, out=buf566)
    del buf564
    del permute_713
    buf567 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf568 = buf559; del buf559  # reuse
    buf569 = buf560; del buf560  # reuse
    buf570 = reinterpret_tensor(buf557, (1, 1024, 512), (524288, 512, 1), 0); del buf557  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_114(c_void_p(buf569.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(add_6.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf570.data_ptr()))
    del add_6
    del getitem_5
    del primals_2
    del rsqrt_1
    buf571 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf570, (512, 1024), (1, 512), 0), view_19, out=buf571)
    del view_19
    buf572 = buf566; del buf566  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf570, (1024, 512), (512, 1), 0), permute_717, out=buf572)
    del permute_717
    buf573 = reinterpret_tensor(buf570, (8, 1024, 64), (65536, 64, 1), 0); del buf570  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_720, reinterpret_tensor(buf572, (8, 1024, 64), (64, 512, 1), 0), out=buf573)
    del permute_720
    buf574 = buf546; del buf546  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf572, (8, 1024, 64), (64, 512, 1), 0), permute_721, out=buf574)
    del permute_721
    buf575 = buf541; del buf541  # reuse
    buf576 = buf574; del buf574  # reuse
    buf577 = reinterpret_tensor(buf542, (8388608, ), (1, ), 0); del buf542  # reuse
    buf580 = reinterpret_tensor(buf176, (8, 1024, 1024), (1048576, 1024, 1), 0); del buf176  # reuse
    buf583 = reinterpret_tensor(buf115, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf115  # reuse
    buf582 = empty((32, 8), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_115(c_void_p(buf576.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(alias_124.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf582.data_ptr()))
    del alias_124
    del buf407
    del buf441
    del buf475
    del buf509
    del buf543
    del buf575
    del buf576
    del buf577
    del getitem_3
    aten.index_put_(buf582, [add_3], buf583, True)
    del add_3
    del buf583
    buf586 = reinterpret_tensor(buf572, (8, 64, 1024), (65536, 1024, 1), 0); del buf572  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_723, buf580, out=buf586)
    del permute_723
    buf587 = reinterpret_tensor(buf554, (8, 1024, 64), (65536, 64, 1), 0); del buf554  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf580, permute_724, out=buf587)
    del buf580
    del permute_724
    buf588 = buf552; del buf552  # reuse
    cpp_fused_view_116(c_void_p(buf573.data_ptr()), c_void_p(buf588.data_ptr()))
    buf589 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf588, (512, 1024), (1, 512), 0), view_1, out=buf589)
    buf590 = reinterpret_tensor(buf573, (1024, 512), (512, 1), 0); del buf573  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf588, permute_729, out=buf590)
    del permute_729
    buf591 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (512, 1024), (1024, 1), 0), view_1, out=buf591)
    buf592 = buf588; del buf588  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (1024, 512), (1, 1024), 0), permute_734, out=buf592)
    del permute_734
    buf593 = reinterpret_tensor(buf586, (1024, 512), (512, 1), 0); del buf586  # reuse
    cpp_fused_view_117(c_void_p(buf587.data_ptr()), c_void_p(buf593.data_ptr()))
    buf594 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf593, (512, 1024), (1, 512), 0), view_1, out=buf594)
    del view_1
    buf595 = reinterpret_tensor(buf587, (1024, 512), (512, 1), 0); del buf587  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf593, permute_739, out=buf595)
    del buf593
    del permute_739
    buf596 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf597 = buf568; del buf568  # reuse
    buf598 = buf569; del buf569  # reuse
    buf600 = buf598; del buf598  # reuse
    buf599 = empty((32128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_118(c_void_p(buf600.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(getitem.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf599.data_ptr()))
    del buf590
    del buf592
    del buf595
    del buf597
    del getitem
    del getitem_1
    del primals_1
    del rsqrt
    aten.index_put_(buf599, [view], buf600, True)
    del buf600
    del view
    buf386 = empty((32128, 512), device='cpu', dtype=torch.float32)
    buf603 = buf386; del buf386  # reuse
    cpp_fused_add_119(c_void_p(buf603.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf599.data_ptr()))
    return (reinterpret_tensor(buf596, (512, ), (1, ), 0), reinterpret_tensor(buf567, (512, ), (1, ), 0), reinterpret_tensor(buf558, (512, ), (1, ), 0), reinterpret_tensor(buf533, (512, ), (1, ), 0), reinterpret_tensor(buf524, (512, ), (1, ), 0), reinterpret_tensor(buf499, (512, ), (1, ), 0), reinterpret_tensor(buf490, (512, ), (1, ), 0), reinterpret_tensor(buf465, (512, ), (1, ), 0), reinterpret_tensor(buf456, (512, ), (1, ), 0), reinterpret_tensor(buf431, (512, ), (1, ), 0), reinterpret_tensor(buf422, (512, ), (1, ), 0), reinterpret_tensor(buf397, (512, ), (1, ), 0), reinterpret_tensor(buf388, (512, ), (1, ), 0), reinterpret_tensor(buf380, (512, ), (1, ), 0), reinterpret_tensor(buf350, (512, ), (1, ), 0), reinterpret_tensor(buf324, (512, ), (1, ), 0), reinterpret_tensor(buf315, (512, ), (1, ), 0), reinterpret_tensor(buf289, (512, ), (1, ), 0), reinterpret_tensor(buf263, (512, ), (1, ), 0), reinterpret_tensor(buf254, (512, ), (1, ), 0), reinterpret_tensor(buf228, (512, ), (1, ), 0), reinterpret_tensor(buf201, (512, ), (1, ), 0), reinterpret_tensor(buf192, (512, ), (1, ), 0), reinterpret_tensor(buf166, (512, ), (1, ), 0), reinterpret_tensor(buf140, (512, ), (1, ), 0), reinterpret_tensor(buf131, (512, ), (1, ), 0), reinterpret_tensor(buf105, (512, ), (1, ), 0), reinterpret_tensor(buf79, (512, ), (1, ), 0), reinterpret_tensor(buf70, (512, ), (1, ), 0), reinterpret_tensor(buf44, (512, ), (1, ), 0), reinterpret_tensor(buf17, (512, ), (1, ), 0), reinterpret_tensor(buf8, (512, ), (1, ), 0), buf603, reinterpret_tensor(buf594, (512, 512), (512, 1), 0), reinterpret_tensor(buf591, (512, 512), (512, 1), 0), reinterpret_tensor(buf589, (512, 512), (512, 1), 0), buf582, reinterpret_tensor(buf571, (512, 512), (512, 1), 0), reinterpret_tensor(buf565, (2048, 512), (512, 1), 0), reinterpret_tensor(buf562, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf556, (512, 512), (512, 1), 0), reinterpret_tensor(buf553, (512, 512), (512, 1), 0), reinterpret_tensor(buf551, (512, 512), (512, 1), 0), reinterpret_tensor(buf537, (512, 512), (512, 1), 0), reinterpret_tensor(buf531, (2048, 512), (512, 1), 0), reinterpret_tensor(buf528, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf522, (512, 512), (512, 1), 0), reinterpret_tensor(buf519, (512, 512), (512, 1), 0), reinterpret_tensor(buf517, (512, 512), (512, 1), 0), reinterpret_tensor(buf503, (512, 512), (512, 1), 0), reinterpret_tensor(buf497, (2048, 512), (512, 1), 0), reinterpret_tensor(buf494, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf488, (512, 512), (512, 1), 0), reinterpret_tensor(buf485, (512, 512), (512, 1), 0), reinterpret_tensor(buf483, (512, 512), (512, 1), 0), reinterpret_tensor(buf469, (512, 512), (512, 1), 0), reinterpret_tensor(buf463, (2048, 512), (512, 1), 0), reinterpret_tensor(buf460, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf454, (512, 512), (512, 1), 0), reinterpret_tensor(buf451, (512, 512), (512, 1), 0), reinterpret_tensor(buf449, (512, 512), (512, 1), 0), reinterpret_tensor(buf435, (512, 512), (512, 1), 0), reinterpret_tensor(buf429, (2048, 512), (512, 1), 0), reinterpret_tensor(buf426, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf420, (512, 512), (512, 1), 0), reinterpret_tensor(buf417, (512, 512), (512, 1), 0), reinterpret_tensor(buf415, (512, 512), (512, 1), 0), reinterpret_tensor(buf401, (512, 512), (512, 1), 0), reinterpret_tensor(buf395, (2048, 512), (512, 1), 0), reinterpret_tensor(buf392, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf378, (512, 512), (512, 1), 0), reinterpret_tensor(buf375, (512, 512), (512, 1), 0), reinterpret_tensor(buf372, (512, 512), (512, 1), 0), buf365, reinterpret_tensor(buf354, (512, 512), (512, 1), 0), reinterpret_tensor(buf348, (512, 512), (512, 1), 0), reinterpret_tensor(buf345, (512, 512), (512, 1), 0), reinterpret_tensor(buf342, (512, 512), (512, 1), 0), reinterpret_tensor(buf328, (512, 512), (512, 1), 0), reinterpret_tensor(buf322, (2048, 512), (512, 1), 0), reinterpret_tensor(buf319, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf313, (512, 512), (512, 1), 0), reinterpret_tensor(buf310, (512, 512), (512, 1), 0), reinterpret_tensor(buf307, (512, 512), (512, 1), 0), reinterpret_tensor(buf293, (512, 512), (512, 1), 0), reinterpret_tensor(buf287, (512, 512), (512, 1), 0), reinterpret_tensor(buf284, (512, 512), (512, 1), 0), reinterpret_tensor(buf281, (512, 512), (512, 1), 0), reinterpret_tensor(buf267, (512, 512), (512, 1), 0), reinterpret_tensor(buf261, (2048, 512), (512, 1), 0), reinterpret_tensor(buf258, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf252, (512, 512), (512, 1), 0), reinterpret_tensor(buf249, (512, 512), (512, 1), 0), reinterpret_tensor(buf246, (512, 512), (512, 1), 0), reinterpret_tensor(buf232, (512, 512), (512, 1), 0), reinterpret_tensor(buf226, (512, 512), (512, 1), 0), reinterpret_tensor(buf222, (512, 512), (512, 1), 0), reinterpret_tensor(buf219, (512, 512), (512, 1), 0), reinterpret_tensor(buf205, (512, 512), (512, 1), 0), reinterpret_tensor(buf199, (2048, 512), (512, 1), 0), reinterpret_tensor(buf196, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf190, (512, 512), (512, 1), 0), reinterpret_tensor(buf187, (512, 512), (512, 1), 0), reinterpret_tensor(buf184, (512, 512), (512, 1), 0), reinterpret_tensor(buf170, (512, 512), (512, 1), 0), reinterpret_tensor(buf164, (512, 512), (512, 1), 0), reinterpret_tensor(buf161, (512, 512), (512, 1), 0), reinterpret_tensor(buf158, (512, 512), (512, 1), 0), reinterpret_tensor(buf144, (512, 512), (512, 1), 0), reinterpret_tensor(buf138, (2048, 512), (512, 1), 0), reinterpret_tensor(buf135, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf129, (512, 512), (512, 1), 0), reinterpret_tensor(buf126, (512, 512), (512, 1), 0), reinterpret_tensor(buf123, (512, 512), (512, 1), 0), reinterpret_tensor(buf109, (512, 512), (512, 1), 0), reinterpret_tensor(buf103, (512, 512), (512, 1), 0), reinterpret_tensor(buf100, (512, 512), (512, 1), 0), reinterpret_tensor(buf97, (512, 512), (512, 1), 0), reinterpret_tensor(buf83, (512, 512), (512, 1), 0), reinterpret_tensor(buf77, (2048, 512), (512, 1), 0), reinterpret_tensor(buf74, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf68, (512, 512), (512, 1), 0), reinterpret_tensor(buf65, (512, 512), (512, 1), 0), reinterpret_tensor(buf62, (512, 512), (512, 1), 0), reinterpret_tensor(buf48, (512, 512), (512, 1), 0), reinterpret_tensor(buf42, (512, 512), (512, 1), 0), reinterpret_tensor(buf39, (512, 512), (512, 1), 0), reinterpret_tensor(buf36, (512, 512), (512, 1), 0), reinterpret_tensor(buf21, (512, 512), (512, 1), 0), reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(buf12, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf6, (32128, 512), (512, 1), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    getitem = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    rsqrt = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_3 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    getitem_3 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_19 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_6 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_23 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_8 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_2 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_43 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_11 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_15 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_47 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_13 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_4 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_67 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_16 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_5 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_71 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_18 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_91 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_21 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_7 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_95 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_23 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_8 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_115 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_26 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_9 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_119 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_28 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_10 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_121 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_139 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_45 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_31 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_11 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_143 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_33 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_12 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    view_145 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    getitem_52 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    rsqrt_13 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_146 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    getitem_55 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_164 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_40 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_14 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_166 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_169 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_59 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_184 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_44 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_15 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_186 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_188 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_65 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_46 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_190 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_208 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_69 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_49 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_17 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_210 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_228 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_52 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_230 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_75 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_232 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_54 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_19 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_234 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_252 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_57 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_20 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_254 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_272 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_85 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_60 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_21 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_274 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_276 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_89 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_62 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_22 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_278 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_296 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_93 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_65 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_23 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_298 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_95 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_316 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_68 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_24 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_318 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_320 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_70 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_322 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_340 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_105 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_73 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_26 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_342 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_360 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_109 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_76 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_27 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_362 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_364 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_113 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_78 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_28 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_366 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_384 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_81 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_29 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_386 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view_404 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_84 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_30 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_406 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_123 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    view_408 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_125 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    add_86 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_31 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.bool)
    view_410 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    sub_24 = rand_strided((1024, 32128), (32128, 1), device='cpu', dtype=torch.float32)
    convert_element_type_7 = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_199 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_207 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_67 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_232 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_69 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_2 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_257 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_73 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_272 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_75 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_297 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_3 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_315 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_79 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_330 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_347 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_81 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_4 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_373 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_85 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_402 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_406 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_87 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_407 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_423 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_427 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_5 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_431 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_91 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_441 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_451 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_456 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_463 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_93 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_465 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_466 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_485 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_6 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_489 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_493 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_496 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_497 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_97 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_498 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_499 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_504 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_518 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_521 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_522 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_99 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_524 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_535 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_540 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_544 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_7 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_548 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_552 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_555 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_556 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_104 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_557 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_558 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_563 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_568 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_573 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_577 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_8 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_581 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_585 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_588 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_589 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_108 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_590 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_591 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_596 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_601 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_606 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_610 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_9 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_614 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_618 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_621 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_622 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_112 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_623 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_624 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_629 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_634 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_639 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_643 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_10 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_647 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_651 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_654 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_655 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_116 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_656 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_657 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_662 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_667 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_672 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_676 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_11 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_680 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_684 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_687 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_688 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_120 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_689 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_690 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_695 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_700 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_705 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_709 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_12 = rand_strided((1, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_713 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_717 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_720 = rand_strided((8, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_721 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    alias_124 = rand_strided((1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_723 = rand_strided((8, 64, 1024), (64, 1, 512), device='cpu', dtype=torch.float32)
    permute_724 = rand_strided((8, 1024, 64), (64, 512, 1), device='cpu', dtype=torch.float32)
    permute_729 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_734 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_739 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 1024, 32128), (32899072, 32128, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((1, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_27 = rand_strided((1, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_134, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, getitem_7, view_23, getitem_9, add_8, rsqrt_2, view_25, getitem_11, view_43, getitem_13, add_11, rsqrt_3, view_45, getitem_15, view_47, getitem_17, add_13, rsqrt_4, view_49, getitem_19, view_67, getitem_21, add_16, rsqrt_5, view_69, getitem_23, view_71, getitem_25, add_18, rsqrt_6, view_73, getitem_27, view_91, getitem_29, add_21, rsqrt_7, view_93, getitem_31, view_95, getitem_33, add_23, rsqrt_8, view_97, getitem_35, view_115, getitem_37, add_26, rsqrt_9, view_117, getitem_39, view_119, getitem_41, add_28, rsqrt_10, view_121, getitem_43, view_139, getitem_45, add_31, rsqrt_11, view_141, getitem_47, view_143, getitem_49, add_33, rsqrt_12, getitem_51, view_145, getitem_52, getitem_53, rsqrt_13, view_146, add_37, getitem_55, view_164, getitem_57, add_40, rsqrt_14, view_166, view_169, getitem_59, view_184, getitem_61, add_44, rsqrt_15, view_186, getitem_63, view_188, getitem_65, add_46, rsqrt_16, view_190, getitem_67, view_208, getitem_69, add_49, rsqrt_17, view_210, getitem_71, view_228, getitem_73, add_52, rsqrt_18, view_230, getitem_75, view_232, getitem_77, add_54, rsqrt_19, view_234, getitem_79, view_252, getitem_81, add_57, rsqrt_20, view_254, getitem_83, view_272, getitem_85, add_60, rsqrt_21, view_274, getitem_87, view_276, getitem_89, add_62, rsqrt_22, view_278, getitem_91, view_296, getitem_93, add_65, rsqrt_23, view_298, getitem_95, view_316, getitem_97, add_68, rsqrt_24, view_318, getitem_99, view_320, getitem_101, add_70, rsqrt_25, view_322, getitem_103, view_340, getitem_105, add_73, rsqrt_26, view_342, getitem_107, view_360, getitem_109, add_76, rsqrt_27, view_362, getitem_111, view_364, getitem_113, add_78, rsqrt_28, view_366, getitem_115, view_384, getitem_117, add_81, rsqrt_29, view_386, getitem_119, view_404, getitem_121, add_84, rsqrt_30, view_406, getitem_123, view_408, getitem_125, add_86, rsqrt_31, getitem_127, view_410, sub_24, convert_element_type_7, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_67, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_69, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_73, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_75, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_79, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_81, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_85, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_87, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_91, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_93, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_97, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_99, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_104, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_108, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_112, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_116, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_120, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_124, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('T5Small', benchmark_compiled_module)
