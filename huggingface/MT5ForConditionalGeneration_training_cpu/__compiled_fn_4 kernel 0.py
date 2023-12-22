
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32014336L); x0+=static_cast<long>(8L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(250112L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (250112L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(250112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (250112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (250112L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (250112L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (250112L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (512L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (512L*x1))];
                    auto tmp6 = in_ptr2[static_cast<long>(x0 + (512L*x1))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                    tmp_acc0 = tmp_acc0 + tmp9;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr4[static_cast<long>(x1)];
                    auto tmp8 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    tmp_acc0 = tmp_acc0 + tmp9;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                auto tmp6 = in_ptr4[static_cast<long>(x1)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp17)(tmp17 * tmp20);
                auto tmp22 = decltype(tmp9)(tmp9 + tmp21);
                in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp22;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp6 = in_ptr4[static_cast<long>(x0)];
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr1[static_cast<long>(x0)] = tmp5;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp6 = in_ptr4[static_cast<long>(x0)];
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr1[static_cast<long>(x0)] = tmp5;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_69 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_74 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp6 = in_ptr4[static_cast<long>(x0)];
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr1[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_81 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_clone_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_86 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp6 = in_ptr4[static_cast<long>(x0)];
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr1[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_clone_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_92 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
            auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
            auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
            auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            auto tmp8 = tmp6 + tmp7;
            auto tmp10 = tmp8 + tmp9;
            auto tmp12 = tmp10 + tmp11;
            auto tmp14 = tmp12 + tmp13;
            auto tmp16 = tmp14 + tmp15;
            auto tmp18 = tmp16 + tmp17;
            auto tmp20 = tmp18 + tmp19;
            auto tmp22 = tmp20 + tmp21;
            auto tmp24 = tmp22 + tmp23;
            auto tmp26 = tmp24 + tmp25;
            auto tmp28 = tmp26 + tmp27;
            auto tmp30 = tmp28 + tmp29;
            auto tmp32 = tmp30 + tmp31;
            tmp32.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_mul_sum_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_mul_sum_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_mul_sum_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_sum_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = in_ptr4[static_cast<long>(x0)];
            auto tmp4 = in_ptr5[static_cast<long>(x0)];
            auto tmp6 = in_ptr6[static_cast<long>(x0)];
            auto tmp8 = in_ptr7[static_cast<long>(x0)];
            auto tmp10 = in_ptr8[static_cast<long>(x0)];
            auto tmp12 = in_ptr9[static_cast<long>(x0)];
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
            auto tmp14 = decltype(tmp13)(tmp13 + tmp0);
            auto tmp15 = static_cast<bool>(0);
            auto tmp16 = static_cast<float>(0.0);
            auto tmp17 = tmp15 ? tmp16 : tmp14;
            out_ptr2[static_cast<long>(x0)] = tmp0;
            out_ptr3[static_cast<long>(x0)] = tmp17;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1) + (384L*x1_inner)));
                    }
                }
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_111 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128057344L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_112 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (512L*x1))];
                    auto tmp1 = in_ptr1[static_cast<long>(x0 + (512L*x1))];
                    auto tmp6 = in_ptr2[static_cast<long>(x0 + (512L*x1))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                    tmp_acc0 = tmp_acc0 + tmp9;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr4[static_cast<long>(x1)];
                    auto tmp8 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    tmp_acc0 = tmp_acc0 + tmp9;
                }
                out_ptr1[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                auto tmp6 = in_ptr4[static_cast<long>(x1)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp17)(tmp17 * tmp20);
                auto tmp22 = decltype(tmp9)(tmp9 + tmp21);
                in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp22;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
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
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_118 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_124 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_130 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_136 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_142 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_148 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_154 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp6 = in_ptr2[static_cast<long>(x0)];
                auto tmp9 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = static_cast<float>(0.5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                auto tmp12 = decltype(tmp8)(tmp8 * tmp11);
                auto tmp13 = decltype(tmp5)(tmp5 * tmp12);
                auto tmp15 = decltype(tmp5)(tmp5 * tmp14);
                auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                auto tmp17 = decltype(tmp9)(tmp9 * tmp9);
                auto tmp18 = decltype(tmp10)(tmp10 - tmp17);
                auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                auto tmp20 = static_cast<float>(0.7978845608028654);
                auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                auto tmp22 = static_cast<float>(0.044715);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp25 = static_cast<float>(3.0);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                auto tmp28 = decltype(tmp21)(tmp21 + tmp27);
                auto tmp29 = decltype(tmp15)(tmp15 * tmp11);
                auto tmp30 = decltype(tmp29)(tmp29 * tmp7);
                auto tmp31 = decltype(tmp28)(tmp28 + tmp30);
                out_ptr0[static_cast<long>(x0)] = tmp13;
                in_out_ptr0[static_cast<long>(x0)] = tmp31;
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_dropout_backward_pow_sum_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = tmp2 * tmp6;
                    tmp_acc0_vec = tmp_acc0_vec + tmp7;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 * tmp4;
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp11 = static_cast<float>(-0.5);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = decltype(tmp6)(tmp6 * tmp6);
                auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                auto tmp16 = static_cast<float>(512.0);
                auto tmp17 = tmp15 / tmp16;
                auto tmp19 = static_cast<float>(2.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = at::vec::Vectorized<float>(tmp17);
                auto tmp23 = tmp22 * tmp21;
                auto tmp24 = tmp9 + tmp23;
                tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp10;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = in_ptr4[static_cast<long>(x0)];
            auto tmp4 = in_ptr5[static_cast<long>(x0)];
            auto tmp6 = in_ptr6[static_cast<long>(x0)];
            auto tmp8 = in_ptr7[static_cast<long>(x0)];
            auto tmp10 = in_ptr8[static_cast<long>(x0)];
            auto tmp12 = in_ptr9[static_cast<long>(x0)];
            auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
            auto tmp14 = decltype(tmp13)(tmp13 + tmp0);
            auto tmp15 = static_cast<bool>(0);
            auto tmp16 = static_cast<float>(0.0);
            auto tmp17 = tmp15 ? tmp16 : tmp14;
            out_ptr2[static_cast<long>(x0)] = tmp0;
            out_ptr3[static_cast<long>(x0)] = tmp17;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_view_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_160 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128057344L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128057344L); x0+=static_cast<long>(8L))
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, mm_4, tanh, mm_5, getitem_7, view_25, getitem_9, add_10, rsqrt_2, view_27, getitem_11, view_45, getitem_13, add_13, rsqrt_3, view_47, mm_11, tanh_1, mm_12, getitem_15, view_51, getitem_17, add_17, rsqrt_4, view_53, getitem_19, view_71, getitem_21, add_20, rsqrt_5, view_73, mm_18, tanh_2, mm_19, getitem_23, view_77, getitem_25, add_24, rsqrt_6, view_79, getitem_27, view_97, getitem_29, add_27, rsqrt_7, view_99, mm_25, tanh_3, mm_26, getitem_31, view_103, getitem_33, add_31, rsqrt_8, view_105, getitem_35, view_123, getitem_37, add_34, rsqrt_9, view_125, mm_32, tanh_4, mm_33, getitem_39, view_129, getitem_41, add_38, rsqrt_10, view_131, getitem_43, view_149, getitem_45, add_41, rsqrt_11, view_151, mm_39, tanh_5, mm_40, getitem_47, view_155, getitem_49, add_45, rsqrt_12, view_157, getitem_51, view_175, getitem_53, add_48, rsqrt_13, view_177, mm_46, tanh_6, mm_47, getitem_55, view_181, getitem_57, add_52, rsqrt_14, view_183, getitem_59, view_201, getitem_61, add_55, rsqrt_15, view_203, mm_53, tanh_7, mm_54, getitem_63, view_207, getitem_65, add_59, rsqrt_16, getitem_67, view_209, getitem_68, getitem_69, rsqrt_17, view_210, add_63, getitem_71, view_228, getitem_73, add_66, rsqrt_18, view_230, view_233, getitem_75, view_248, getitem_77, add_70, rsqrt_19, view_250, mm_64, tanh_8, mm_65, getitem_79, view_254, getitem_81, add_74, rsqrt_20, view_256, getitem_83, view_274, getitem_85, add_77, rsqrt_21, view_276, getitem_87, view_294, getitem_89, add_80, rsqrt_22, view_296, mm_75, tanh_9, mm_76, getitem_91, view_300, getitem_93, add_84, rsqrt_23, view_302, getitem_95, view_320, getitem_97, add_87, rsqrt_24, view_322, getitem_99, view_340, getitem_101, add_90, rsqrt_25, view_342, mm_86, tanh_10, mm_87, getitem_103, view_346, getitem_105, add_94, rsqrt_26, view_348, getitem_107, view_366, getitem_109, add_97, rsqrt_27, view_368, getitem_111, view_386, getitem_113, add_100, rsqrt_28, view_388, mm_97, tanh_11, mm_98, getitem_115, view_392, getitem_117, add_104, rsqrt_29, view_394, getitem_119, view_412, getitem_121, add_107, rsqrt_30, view_414, getitem_123, view_432, getitem_125, add_110, rsqrt_31, view_434, mm_108, tanh_12, mm_109, getitem_127, view_438, getitem_129, add_114, rsqrt_32, view_440, getitem_131, view_458, getitem_133, add_117, rsqrt_33, view_460, getitem_135, view_478, getitem_137, add_120, rsqrt_34, view_480, mm_119, tanh_13, mm_120, getitem_139, view_484, getitem_141, add_124, rsqrt_35, view_486, getitem_143, view_504, getitem_145, add_127, rsqrt_36, view_506, getitem_147, view_524, getitem_149, add_130, rsqrt_37, view_526, mm_130, tanh_14, mm_131, getitem_151, view_530, getitem_153, add_134, rsqrt_38, view_532, getitem_155, view_550, getitem_157, add_137, rsqrt_39, view_552, getitem_159, view_570, getitem_161, add_140, rsqrt_40, view_572, mm_141, tanh_15, mm_142, getitem_163, view_576, getitem_165, add_144, rsqrt_41, getitem_167, view_578, sub_30, convert_element_type_7, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, alias_87, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, alias_89, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, alias_93, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, alias_95, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, alias_99, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, alias_101, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, alias_105, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, alias_107, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, alias_111, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, alias_113, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, alias_117, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, alias_119, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, alias_123, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, alias_125, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, alias_129, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, alias_131, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, alias_136, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, alias_140, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, alias_144, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, alias_148, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, alias_152, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, alias_156, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, alias_160, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, alias_164, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35 = args
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
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_192, (1, 128), (128, 1))
    assert_size_stride(view, (1, 128), (128, 1))
    assert_size_stride(getitem, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(getitem_1, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_1, (128, 512), (512, 1))
    assert_size_stride(add_3, (128, 128), (128, 1))
    assert_size_stride(getitem_3, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_19, (128, 384), (384, 1))
    assert_size_stride(getitem_5, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_6, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_1, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_21, (128, 512), (512, 1))
    assert_size_stride(mm_4, (128, 1024), (1024, 1))
    assert_size_stride(tanh, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_5, (128, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_25, (128, 1024), (1024, 1))
    assert_size_stride(getitem_9, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_10, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_27, (128, 512), (512, 1))
    assert_size_stride(getitem_11, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_45, (128, 384), (384, 1))
    assert_size_stride(getitem_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_13, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_3, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_47, (128, 512), (512, 1))
    assert_size_stride(mm_11, (128, 1024), (1024, 1))
    assert_size_stride(tanh_1, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_12, (128, 1024), (1024, 1))
    assert_size_stride(getitem_15, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_51, (128, 1024), (1024, 1))
    assert_size_stride(getitem_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_17, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_4, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_53, (128, 512), (512, 1))
    assert_size_stride(getitem_19, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_71, (128, 384), (384, 1))
    assert_size_stride(getitem_21, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_20, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_5, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_73, (128, 512), (512, 1))
    assert_size_stride(mm_18, (128, 1024), (1024, 1))
    assert_size_stride(tanh_2, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_19, (128, 1024), (1024, 1))
    assert_size_stride(getitem_23, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_77, (128, 1024), (1024, 1))
    assert_size_stride(getitem_25, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_24, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_6, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_79, (128, 512), (512, 1))
    assert_size_stride(getitem_27, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_97, (128, 384), (384, 1))
    assert_size_stride(getitem_29, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_27, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_7, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_99, (128, 512), (512, 1))
    assert_size_stride(mm_25, (128, 1024), (1024, 1))
    assert_size_stride(tanh_3, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_26, (128, 1024), (1024, 1))
    assert_size_stride(getitem_31, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_103, (128, 1024), (1024, 1))
    assert_size_stride(getitem_33, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_31, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_8, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_105, (128, 512), (512, 1))
    assert_size_stride(getitem_35, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_123, (128, 384), (384, 1))
    assert_size_stride(getitem_37, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_34, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_9, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_125, (128, 512), (512, 1))
    assert_size_stride(mm_32, (128, 1024), (1024, 1))
    assert_size_stride(tanh_4, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_33, (128, 1024), (1024, 1))
    assert_size_stride(getitem_39, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_129, (128, 1024), (1024, 1))
    assert_size_stride(getitem_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_38, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_10, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_131, (128, 512), (512, 1))
    assert_size_stride(getitem_43, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_149, (128, 384), (384, 1))
    assert_size_stride(getitem_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_41, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_11, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_151, (128, 512), (512, 1))
    assert_size_stride(mm_39, (128, 1024), (1024, 1))
    assert_size_stride(tanh_5, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_40, (128, 1024), (1024, 1))
    assert_size_stride(getitem_47, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_155, (128, 1024), (1024, 1))
    assert_size_stride(getitem_49, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_45, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_12, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_157, (128, 512), (512, 1))
    assert_size_stride(getitem_51, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_175, (128, 384), (384, 1))
    assert_size_stride(getitem_53, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_48, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_13, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_177, (128, 512), (512, 1))
    assert_size_stride(mm_46, (128, 1024), (1024, 1))
    assert_size_stride(tanh_6, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_47, (128, 1024), (1024, 1))
    assert_size_stride(getitem_55, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_181, (128, 1024), (1024, 1))
    assert_size_stride(getitem_57, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_52, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_14, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_183, (128, 512), (512, 1))
    assert_size_stride(getitem_59, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_201, (128, 384), (384, 1))
    assert_size_stride(getitem_61, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_55, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_15, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_203, (128, 512), (512, 1))
    assert_size_stride(mm_53, (128, 1024), (1024, 1))
    assert_size_stride(tanh_7, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_54, (128, 1024), (1024, 1))
    assert_size_stride(getitem_63, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_207, (128, 1024), (1024, 1))
    assert_size_stride(getitem_65, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_59, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_16, (1, 128, 1), (128, 1, 1))
    assert_size_stride(getitem_67, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_209, (1, 128), (128, 1))
    assert_size_stride(getitem_68, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(getitem_69, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_17, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_210, (128, 512), (512, 1))
    assert_size_stride(add_63, (128, 128), (128, 1))
    assert_size_stride(getitem_71, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_228, (128, 384), (384, 1))
    assert_size_stride(getitem_73, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_66, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_230, (128, 512), (512, 1))
    assert_size_stride(view_233, (128, 512), (512, 1))
    assert_size_stride(getitem_75, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_248, (128, 384), (384, 1))
    assert_size_stride(getitem_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_70, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_250, (128, 512), (512, 1))
    assert_size_stride(mm_64, (128, 1024), (1024, 1))
    assert_size_stride(tanh_8, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_65, (128, 1024), (1024, 1))
    assert_size_stride(getitem_79, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_254, (128, 1024), (1024, 1))
    assert_size_stride(getitem_81, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_74, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_20, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_256, (128, 512), (512, 1))
    assert_size_stride(getitem_83, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_274, (128, 384), (384, 1))
    assert_size_stride(getitem_85, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_77, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_276, (128, 512), (512, 1))
    assert_size_stride(getitem_87, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_294, (128, 384), (384, 1))
    assert_size_stride(getitem_89, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_80, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_296, (128, 512), (512, 1))
    assert_size_stride(mm_75, (128, 1024), (1024, 1))
    assert_size_stride(tanh_9, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_76, (128, 1024), (1024, 1))
    assert_size_stride(getitem_91, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_300, (128, 1024), (1024, 1))
    assert_size_stride(getitem_93, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_84, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_23, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_302, (128, 512), (512, 1))
    assert_size_stride(getitem_95, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_320, (128, 384), (384, 1))
    assert_size_stride(getitem_97, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_87, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_322, (128, 512), (512, 1))
    assert_size_stride(getitem_99, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_340, (128, 384), (384, 1))
    assert_size_stride(getitem_101, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_90, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_342, (128, 512), (512, 1))
    assert_size_stride(mm_86, (128, 1024), (1024, 1))
    assert_size_stride(tanh_10, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_87, (128, 1024), (1024, 1))
    assert_size_stride(getitem_103, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_346, (128, 1024), (1024, 1))
    assert_size_stride(getitem_105, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_94, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_26, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_348, (128, 512), (512, 1))
    assert_size_stride(getitem_107, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_366, (128, 384), (384, 1))
    assert_size_stride(getitem_109, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_97, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_368, (128, 512), (512, 1))
    assert_size_stride(getitem_111, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_386, (128, 384), (384, 1))
    assert_size_stride(getitem_113, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_100, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_388, (128, 512), (512, 1))
    assert_size_stride(mm_97, (128, 1024), (1024, 1))
    assert_size_stride(tanh_11, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_98, (128, 1024), (1024, 1))
    assert_size_stride(getitem_115, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_392, (128, 1024), (1024, 1))
    assert_size_stride(getitem_117, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_104, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_29, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_394, (128, 512), (512, 1))
    assert_size_stride(getitem_119, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_412, (128, 384), (384, 1))
    assert_size_stride(getitem_121, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_107, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_414, (128, 512), (512, 1))
    assert_size_stride(getitem_123, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_432, (128, 384), (384, 1))
    assert_size_stride(getitem_125, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_110, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_434, (128, 512), (512, 1))
    assert_size_stride(mm_108, (128, 1024), (1024, 1))
    assert_size_stride(tanh_12, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_109, (128, 1024), (1024, 1))
    assert_size_stride(getitem_127, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_438, (128, 1024), (1024, 1))
    assert_size_stride(getitem_129, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_114, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_32, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_440, (128, 512), (512, 1))
    assert_size_stride(getitem_131, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_458, (128, 384), (384, 1))
    assert_size_stride(getitem_133, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_117, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_460, (128, 512), (512, 1))
    assert_size_stride(getitem_135, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_478, (128, 384), (384, 1))
    assert_size_stride(getitem_137, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_120, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_480, (128, 512), (512, 1))
    assert_size_stride(mm_119, (128, 1024), (1024, 1))
    assert_size_stride(tanh_13, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_120, (128, 1024), (1024, 1))
    assert_size_stride(getitem_139, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_484, (128, 1024), (1024, 1))
    assert_size_stride(getitem_141, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_124, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_35, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_486, (128, 512), (512, 1))
    assert_size_stride(getitem_143, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_504, (128, 384), (384, 1))
    assert_size_stride(getitem_145, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_127, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_506, (128, 512), (512, 1))
    assert_size_stride(getitem_147, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_524, (128, 384), (384, 1))
    assert_size_stride(getitem_149, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_130, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_37, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_526, (128, 512), (512, 1))
    assert_size_stride(mm_130, (128, 1024), (1024, 1))
    assert_size_stride(tanh_14, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_131, (128, 1024), (1024, 1))
    assert_size_stride(getitem_151, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_530, (128, 1024), (1024, 1))
    assert_size_stride(getitem_153, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_134, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_38, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_532, (128, 512), (512, 1))
    assert_size_stride(getitem_155, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_550, (128, 384), (384, 1))
    assert_size_stride(getitem_157, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_137, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_39, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_552, (128, 512), (512, 1))
    assert_size_stride(getitem_159, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(view_570, (128, 384), (384, 1))
    assert_size_stride(getitem_161, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_140, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_40, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_572, (128, 512), (512, 1))
    assert_size_stride(mm_141, (128, 1024), (1024, 1))
    assert_size_stride(tanh_15, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(mm_142, (128, 1024), (1024, 1))
    assert_size_stride(getitem_163, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(view_576, (128, 1024), (1024, 1))
    assert_size_stride(getitem_165, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(add_144, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(rsqrt_41, (1, 128, 1), (128, 1, 1))
    assert_size_stride(getitem_167, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(view_578, (128, 512), (512, 1))
    assert_size_stride(sub_30, (128, 250112), (250112, 1))
    assert_size_stride(convert_element_type_7, (), ())
    assert_size_stride(permute_269, (250112, 512), (512, 1))
    assert_size_stride(permute_273, (512, 1024), (1024, 1))
    assert_size_stride(permute_277, (1024, 512), (512, 1))
    assert_size_stride(permute_281, (1024, 512), (512, 1))
    assert_size_stride(permute_285, (512, 384), (384, 1))
    assert_size_stride(permute_288, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_289, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_87, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_290, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_291, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_296, (384, 512), (512, 1))
    assert_size_stride(permute_301, (384, 512), (512, 1))
    assert_size_stride(permute_306, (384, 512), (512, 1))
    assert_size_stride(permute_310, (512, 384), (384, 1))
    assert_size_stride(permute_313, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_314, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_89, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_315, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_316, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_321, (384, 512), (512, 1))
    assert_size_stride(permute_326, (384, 512), (512, 1))
    assert_size_stride(permute_331, (384, 512), (512, 1))
    assert_size_stride(permute_335, (512, 1024), (1024, 1))
    assert_size_stride(permute_339, (1024, 512), (512, 1))
    assert_size_stride(permute_343, (1024, 512), (512, 1))
    assert_size_stride(permute_347, (512, 384), (384, 1))
    assert_size_stride(permute_350, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_351, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_93, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_352, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_353, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_358, (384, 512), (512, 1))
    assert_size_stride(permute_363, (384, 512), (512, 1))
    assert_size_stride(permute_368, (384, 512), (512, 1))
    assert_size_stride(permute_372, (512, 384), (384, 1))
    assert_size_stride(permute_375, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_376, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_95, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_377, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_378, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_383, (384, 512), (512, 1))
    assert_size_stride(permute_388, (384, 512), (512, 1))
    assert_size_stride(permute_393, (384, 512), (512, 1))
    assert_size_stride(permute_397, (512, 1024), (1024, 1))
    assert_size_stride(permute_401, (1024, 512), (512, 1))
    assert_size_stride(permute_405, (1024, 512), (512, 1))
    assert_size_stride(permute_409, (512, 384), (384, 1))
    assert_size_stride(permute_412, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_413, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_99, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_414, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_415, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_420, (384, 512), (512, 1))
    assert_size_stride(permute_425, (384, 512), (512, 1))
    assert_size_stride(permute_430, (384, 512), (512, 1))
    assert_size_stride(permute_434, (512, 384), (384, 1))
    assert_size_stride(permute_437, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_438, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_101, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_439, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_440, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_445, (384, 512), (512, 1))
    assert_size_stride(permute_450, (384, 512), (512, 1))
    assert_size_stride(permute_455, (384, 512), (512, 1))
    assert_size_stride(permute_459, (512, 1024), (1024, 1))
    assert_size_stride(permute_463, (1024, 512), (512, 1))
    assert_size_stride(permute_467, (1024, 512), (512, 1))
    assert_size_stride(permute_471, (512, 384), (384, 1))
    assert_size_stride(permute_474, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_475, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_105, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_476, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_477, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_482, (384, 512), (512, 1))
    assert_size_stride(permute_487, (384, 512), (512, 1))
    assert_size_stride(permute_492, (384, 512), (512, 1))
    assert_size_stride(permute_496, (512, 384), (384, 1))
    assert_size_stride(permute_499, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_500, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_107, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_501, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_502, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_507, (384, 512), (512, 1))
    assert_size_stride(permute_512, (384, 512), (512, 1))
    assert_size_stride(permute_517, (384, 512), (512, 1))
    assert_size_stride(permute_521, (512, 1024), (1024, 1))
    assert_size_stride(permute_525, (1024, 512), (512, 1))
    assert_size_stride(permute_529, (1024, 512), (512, 1))
    assert_size_stride(permute_533, (512, 384), (384, 1))
    assert_size_stride(permute_536, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_537, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_111, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_538, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_539, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_544, (384, 512), (512, 1))
    assert_size_stride(permute_549, (384, 512), (512, 1))
    assert_size_stride(permute_554, (384, 512), (512, 1))
    assert_size_stride(permute_558, (512, 384), (384, 1))
    assert_size_stride(permute_561, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_562, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_113, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_563, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_564, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_569, (384, 512), (512, 1))
    assert_size_stride(permute_574, (384, 512), (512, 1))
    assert_size_stride(permute_579, (384, 512), (512, 1))
    assert_size_stride(permute_583, (512, 1024), (1024, 1))
    assert_size_stride(permute_587, (1024, 512), (512, 1))
    assert_size_stride(permute_591, (1024, 512), (512, 1))
    assert_size_stride(permute_595, (512, 384), (384, 1))
    assert_size_stride(permute_598, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_599, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_117, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_600, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_601, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_606, (384, 512), (512, 1))
    assert_size_stride(permute_611, (384, 512), (512, 1))
    assert_size_stride(permute_616, (384, 512), (512, 1))
    assert_size_stride(permute_620, (512, 384), (384, 1))
    assert_size_stride(permute_623, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_624, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_119, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_625, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_626, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_631, (384, 512), (512, 1))
    assert_size_stride(permute_636, (384, 512), (512, 1))
    assert_size_stride(permute_641, (384, 512), (512, 1))
    assert_size_stride(permute_645, (512, 1024), (1024, 1))
    assert_size_stride(permute_649, (1024, 512), (512, 1))
    assert_size_stride(permute_653, (1024, 512), (512, 1))
    assert_size_stride(permute_657, (512, 384), (384, 1))
    assert_size_stride(permute_660, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_661, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_123, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_662, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_663, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_668, (384, 512), (512, 1))
    assert_size_stride(permute_673, (384, 512), (512, 1))
    assert_size_stride(permute_678, (384, 512), (512, 1))
    assert_size_stride(permute_682, (512, 384), (384, 1))
    assert_size_stride(permute_685, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_686, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_125, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_687, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_688, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_693, (384, 512), (512, 1))
    assert_size_stride(permute_698, (384, 512), (512, 1))
    assert_size_stride(permute_703, (384, 512), (512, 1))
    assert_size_stride(permute_707, (512, 1024), (1024, 1))
    assert_size_stride(permute_711, (1024, 512), (512, 1))
    assert_size_stride(permute_715, (1024, 512), (512, 1))
    assert_size_stride(permute_719, (512, 384), (384, 1))
    assert_size_stride(permute_722, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_723, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_129, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_724, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_725, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_730, (384, 512), (512, 1))
    assert_size_stride(permute_735, (384, 512), (512, 1))
    assert_size_stride(permute_740, (384, 512), (512, 1))
    assert_size_stride(permute_744, (512, 384), (384, 1))
    assert_size_stride(permute_747, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_748, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_131, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_750, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_751, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_756, (384, 512), (512, 1))
    assert_size_stride(permute_761, (384, 512), (512, 1))
    assert_size_stride(permute_766, (384, 512), (512, 1))
    assert_size_stride(permute_770, (512, 1024), (1024, 1))
    assert_size_stride(permute_774, (1024, 512), (512, 1))
    assert_size_stride(permute_778, (1024, 512), (512, 1))
    assert_size_stride(permute_782, (512, 384), (384, 1))
    assert_size_stride(permute_785, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_786, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_136, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_787, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_788, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_793, (384, 512), (512, 1))
    assert_size_stride(permute_798, (384, 512), (512, 1))
    assert_size_stride(permute_803, (384, 512), (512, 1))
    assert_size_stride(permute_807, (512, 1024), (1024, 1))
    assert_size_stride(permute_811, (1024, 512), (512, 1))
    assert_size_stride(permute_815, (1024, 512), (512, 1))
    assert_size_stride(permute_819, (512, 384), (384, 1))
    assert_size_stride(permute_822, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_823, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_140, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_824, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_825, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_830, (384, 512), (512, 1))
    assert_size_stride(permute_835, (384, 512), (512, 1))
    assert_size_stride(permute_840, (384, 512), (512, 1))
    assert_size_stride(permute_844, (512, 1024), (1024, 1))
    assert_size_stride(permute_848, (1024, 512), (512, 1))
    assert_size_stride(permute_852, (1024, 512), (512, 1))
    assert_size_stride(permute_856, (512, 384), (384, 1))
    assert_size_stride(permute_859, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_860, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_144, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_861, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_862, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_867, (384, 512), (512, 1))
    assert_size_stride(permute_872, (384, 512), (512, 1))
    assert_size_stride(permute_877, (384, 512), (512, 1))
    assert_size_stride(permute_881, (512, 1024), (1024, 1))
    assert_size_stride(permute_885, (1024, 512), (512, 1))
    assert_size_stride(permute_889, (1024, 512), (512, 1))
    assert_size_stride(permute_893, (512, 384), (384, 1))
    assert_size_stride(permute_896, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_897, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_148, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_898, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_899, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_904, (384, 512), (512, 1))
    assert_size_stride(permute_909, (384, 512), (512, 1))
    assert_size_stride(permute_914, (384, 512), (512, 1))
    assert_size_stride(permute_918, (512, 1024), (1024, 1))
    assert_size_stride(permute_922, (1024, 512), (512, 1))
    assert_size_stride(permute_926, (1024, 512), (512, 1))
    assert_size_stride(permute_930, (512, 384), (384, 1))
    assert_size_stride(permute_933, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_934, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_152, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_935, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_936, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_941, (384, 512), (512, 1))
    assert_size_stride(permute_946, (384, 512), (512, 1))
    assert_size_stride(permute_951, (384, 512), (512, 1))
    assert_size_stride(permute_955, (512, 1024), (1024, 1))
    assert_size_stride(permute_959, (1024, 512), (512, 1))
    assert_size_stride(permute_963, (1024, 512), (512, 1))
    assert_size_stride(permute_967, (512, 384), (384, 1))
    assert_size_stride(permute_970, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_971, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_156, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_972, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_973, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_978, (384, 512), (512, 1))
    assert_size_stride(permute_983, (384, 512), (512, 1))
    assert_size_stride(permute_988, (384, 512), (512, 1))
    assert_size_stride(permute_992, (512, 1024), (1024, 1))
    assert_size_stride(permute_996, (1024, 512), (512, 1))
    assert_size_stride(permute_1000, (1024, 512), (512, 1))
    assert_size_stride(permute_1004, (512, 384), (384, 1))
    assert_size_stride(permute_1007, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1008, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_160, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_1009, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_1010, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_1015, (384, 512), (512, 1))
    assert_size_stride(permute_1020, (384, 512), (512, 1))
    assert_size_stride(permute_1025, (384, 512), (512, 1))
    assert_size_stride(permute_1029, (512, 1024), (1024, 1))
    assert_size_stride(permute_1033, (1024, 512), (512, 1))
    assert_size_stride(permute_1037, (1024, 512), (512, 1))
    assert_size_stride(permute_1041, (512, 384), (384, 1))
    assert_size_stride(permute_1044, (6, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_1045, (6, 64, 128), (64, 1, 384))
    assert_size_stride(alias_164, (1, 6, 128, 128), (98304, 16384, 128, 1))
    assert_size_stride(permute_1047, (6, 64, 128), (64, 1, 384))
    assert_size_stride(permute_1048, (6, 128, 64), (64, 384, 1))
    assert_size_stride(permute_1053, (384, 512), (512, 1))
    assert_size_stride(permute_1058, (384, 512), (512, 1))
    assert_size_stride(permute_1063, (384, 512), (512, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128, 250112), (32014336, 250112, 1))
    assert_size_stride(tangents_3, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_4, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_5, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_6, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_7, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_8, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_9, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_10, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_11, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_12, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_13, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_14, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_15, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_16, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_17, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_18, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_19, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_20, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_21, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_22, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_23, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_24, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_25, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_26, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_27, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_28, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_29, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_30, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_31, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_32, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_33, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_34, (1, 6, 128, 64), (49152, 8192, 64, 1))
    assert_size_stride(tangents_35, (1, 128, 512), (65536, 512, 1))
    buf0 = empty((128, 250112), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_192.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((128, 250112), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 128, 250112), (32014336, 250112, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type_7.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_30.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type_7
    del primals_192
    del sub_30
    del tangents_1
    del tangents_2
    buf6 = empty((250112, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (250112, 128), (1, 250112), 0), view_578, out=buf6)
    del view_578
    buf7 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (128, 250112), (250112, 1), 0), permute_269, out=buf7)
    del buf5
    del permute_269
    buf8 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 128), 0); del buf4  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 128, 512), (65536, 512, 1), 0); del buf7  # reuse
    buf11 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_2(c_void_p(buf10.data_ptr()), c_void_p(getitem_167.data_ptr()), c_void_p(add_144.data_ptr()), c_void_p(rsqrt_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(getitem_165.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    del add_144
    del getitem_165
    del getitem_167
    del primals_42
    del rsqrt_41
    buf12 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (512, 128), (1, 512), 0), view_576, out=buf12)
    del view_576
    buf13 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (128, 512), (512, 1), 0), permute_273, out=buf13)
    del permute_273
    buf14 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf18 = buf17; del buf17  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_3(c_void_p(buf18.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(getitem_163.data_ptr()), c_void_p(mm_141.data_ptr()), c_void_p(tanh_15.data_ptr()), c_void_p(mm_142.data_ptr()), c_void_p(buf14.data_ptr()))
    del getitem_163
    del mm_141
    del mm_142
    del tanh_15
    buf15 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (1024, 128), (1, 1024), 0), view_572, out=buf15)
    buf16 = reinterpret_tensor(buf11, (128, 512), (512, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (128, 1024), (1024, 1), 0), permute_277, out=buf16)
    del permute_277
    buf19 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 128), (1, 1024), 0), view_572, out=buf19)
    del view_572
    buf20 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (128, 1024), (1024, 1), 0), permute_281, out=buf20)
    del permute_281
    buf21 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf22 = buf9; del buf9  # reuse
    buf23 = buf10; del buf10  # reuse
    buf24 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_4(c_void_p(buf23.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(add_140.data_ptr()), c_void_p(rsqrt_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(getitem_161.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    del add_140
    del getitem_161
    del primals_41
    del rsqrt_40
    buf25 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (512, 128), (1, 512), 0), view_570, out=buf25)
    del view_570
    buf26 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (128, 512), (512, 1), 0), permute_285, out=buf26)
    del permute_285
    buf27 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_288, reinterpret_tensor(buf26, (6, 128, 64), (64, 384, 1), 0), out=buf27)
    del permute_288
    buf28 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (6, 128, 64), (64, 384, 1), 0), permute_289, out=buf28)
    del permute_289
    buf29 = empty_strided((1, 6, 128, 1), (768, 128, 1, 768), device='cpu', dtype=torch.float32)
    buf30 = buf28; del buf28  # reuse
    buf32 = empty((98304, ), device='cpu', dtype=torch.float32)
    buf35 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_5(c_void_p(buf30.data_ptr()), c_void_p(getitem_159.data_ptr()), c_void_p(alias_87.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf35.data_ptr()))
    del alias_87
    del getitem_159
    buf37 = reinterpret_tensor(buf26, (6, 64, 128), (8192, 128, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_290, buf35, out=buf37)
    del permute_290
    buf38 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf35, permute_291, out=buf38)
    del permute_291
    buf39 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(tangents_34.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf39.data_ptr()))
    del tangents_34
    buf40 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (384, 128), (1, 384), 0), view_233, out=buf40)
    buf41 = reinterpret_tensor(buf24, (128, 512), (512, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (128, 384), (384, 1), 0), permute_296, out=buf41)
    del permute_296
    buf42 = buf39; del buf39  # reuse
    cpp_fused_clone_7(c_void_p(tangents_33.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf42.data_ptr()))
    del tangents_33
    buf43 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (384, 128), (1, 384), 0), view_233, out=buf43)
    buf44 = buf20; del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (128, 384), (384, 1), 0), permute_301, out=buf44)
    del permute_301
    buf45 = reinterpret_tensor(buf42, (128, 384), (384, 1), 0); del buf42  # reuse
    cpp_fused_view_8(c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (384, 128), (1, 384), 0), view_552, out=buf46)
    del view_552
    buf47 = buf16; del buf16  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_306, out=buf47)
    del permute_306
    buf48 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf49 = buf22; del buf22  # reuse
    buf50 = buf23; del buf23  # reuse
    buf51 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_9(c_void_p(buf50.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(add_137.data_ptr()), c_void_p(rsqrt_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(getitem_157.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()))
    del add_137
    del getitem_157
    del primals_40
    del rsqrt_39
    buf52 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (512, 128), (1, 512), 0), view_550, out=buf52)
    del view_550
    buf53 = buf45; del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), permute_310, out=buf53)
    del permute_310
    buf54 = buf38; del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_313, reinterpret_tensor(buf53, (6, 128, 64), (64, 384, 1), 0), out=buf54)
    del permute_313
    buf55 = buf35; del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (6, 128, 64), (64, 384, 1), 0), permute_314, out=buf55)
    del permute_314
    buf56 = buf29; del buf29  # reuse
    buf57 = buf55; del buf55  # reuse
    buf58 = buf32; del buf32  # reuse
    buf61 = buf30; del buf30  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_10(c_void_p(buf57.data_ptr()), c_void_p(getitem_155.data_ptr()), c_void_p(alias_89.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_89
    del getitem_155
    buf63 = reinterpret_tensor(buf53, (6, 64, 128), (8192, 128, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_315, buf61, out=buf63)
    del permute_315
    buf64 = reinterpret_tensor(buf37, (6, 128, 64), (8192, 64, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf61, permute_316, out=buf64)
    del permute_316
    buf65 = reinterpret_tensor(buf27, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf27  # reuse
    cpp_fused_clone_11(c_void_p(tangents_32.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf65.data_ptr()))
    del tangents_32
    buf66 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (384, 128), (1, 384), 0), view_532, out=buf66)
    buf67 = reinterpret_tensor(buf51, (128, 512), (512, 1), 0); del buf51  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (128, 384), (384, 1), 0), permute_321, out=buf67)
    del permute_321
    buf68 = buf65; del buf65  # reuse
    cpp_fused_clone_12(c_void_p(tangents_31.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf68.data_ptr()))
    del tangents_31
    buf69 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (384, 128), (1, 384), 0), view_532, out=buf69)
    buf70 = buf47; del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (128, 384), (384, 1), 0), permute_326, out=buf70)
    del permute_326
    buf71 = reinterpret_tensor(buf68, (128, 384), (384, 1), 0); del buf68  # reuse
    cpp_fused_view_13(c_void_p(buf64.data_ptr()), c_void_p(buf71.data_ptr()))
    buf72 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (384, 128), (1, 384), 0), view_532, out=buf72)
    del view_532
    buf73 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf71, permute_331, out=buf73)
    del permute_331
    buf74 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf75 = buf49; del buf49  # reuse
    buf76 = buf50; del buf50  # reuse
    buf77 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_14(c_void_p(buf76.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(add_134.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(getitem_153.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del add_134
    del getitem_153
    del primals_39
    del rsqrt_38
    buf78 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (512, 128), (1, 512), 0), view_530, out=buf78)
    del view_530
    buf79 = reinterpret_tensor(buf18, (128, 1024), (1024, 1), 0); del buf18  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (128, 512), (512, 1), 0), permute_335, out=buf79)
    del permute_335
    buf80 = buf14; del buf14  # reuse
    buf83 = reinterpret_tensor(buf13, (1, 128, 1024), (131072, 1024, 1), 0); del buf13  # reuse
    buf84 = buf83; del buf83  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_15(c_void_p(buf84.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(getitem_151.data_ptr()), c_void_p(mm_130.data_ptr()), c_void_p(tanh_14.data_ptr()), c_void_p(mm_131.data_ptr()), c_void_p(buf80.data_ptr()))
    del getitem_151
    del mm_130
    del mm_131
    del tanh_14
    buf81 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (1024, 128), (1, 1024), 0), view_526, out=buf81)
    buf82 = reinterpret_tensor(buf77, (128, 512), (512, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (128, 1024), (1024, 1), 0), permute_339, out=buf82)
    del permute_339
    buf85 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (1024, 128), (1, 1024), 0), view_526, out=buf85)
    del view_526
    buf86 = buf73; del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (128, 1024), (1024, 1), 0), permute_343, out=buf86)
    del permute_343
    buf87 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf88 = buf75; del buf75  # reuse
    buf89 = buf76; del buf76  # reuse
    buf90 = reinterpret_tensor(buf70, (1, 128, 512), (65536, 512, 1), 0); del buf70  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_16(c_void_p(buf89.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(add_130.data_ptr()), c_void_p(rsqrt_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del add_130
    del getitem_149
    del primals_38
    del rsqrt_37
    buf91 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (512, 128), (1, 512), 0), view_524, out=buf91)
    del view_524
    buf92 = buf71; del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (128, 512), (512, 1), 0), permute_347, out=buf92)
    del permute_347
    buf93 = buf64; del buf64  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_350, reinterpret_tensor(buf92, (6, 128, 64), (64, 384, 1), 0), out=buf93)
    del permute_350
    buf94 = buf61; del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf92, (6, 128, 64), (64, 384, 1), 0), permute_351, out=buf94)
    del permute_351
    buf95 = buf56; del buf56  # reuse
    buf96 = buf94; del buf94  # reuse
    buf97 = reinterpret_tensor(buf57, (98304, ), (1, ), 0); del buf57  # reuse
    buf100 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_17(c_void_p(buf96.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(alias_93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf100.data_ptr()))
    del alias_93
    del getitem_147
    buf102 = reinterpret_tensor(buf92, (6, 64, 128), (8192, 128, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_352, buf100, out=buf102)
    del permute_352
    buf103 = reinterpret_tensor(buf63, (6, 128, 64), (8192, 64, 1), 0); del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf100, permute_353, out=buf103)
    del permute_353
    buf104 = reinterpret_tensor(buf54, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf54  # reuse
    cpp_fused_clone_18(c_void_p(tangents_30.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf104.data_ptr()))
    del tangents_30
    buf105 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (384, 128), (1, 384), 0), view_233, out=buf105)
    buf106 = reinterpret_tensor(buf90, (128, 512), (512, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (128, 384), (384, 1), 0), permute_358, out=buf106)
    del permute_358
    buf107 = buf104; del buf104  # reuse
    cpp_fused_clone_19(c_void_p(tangents_29.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf107.data_ptr()))
    del tangents_29
    buf108 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (384, 128), (1, 384), 0), view_233, out=buf108)
    buf109 = buf86; del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (128, 384), (384, 1), 0), permute_363, out=buf109)
    del permute_363
    buf110 = reinterpret_tensor(buf107, (128, 384), (384, 1), 0); del buf107  # reuse
    cpp_fused_view_20(c_void_p(buf103.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (384, 128), (1, 384), 0), view_506, out=buf111)
    del view_506
    buf112 = buf82; del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf110, permute_368, out=buf112)
    del permute_368
    buf113 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf114 = buf88; del buf88  # reuse
    buf115 = reinterpret_tensor(buf112, (1, 128, 512), (65536, 512, 1), 0); del buf112  # reuse
    buf116 = reinterpret_tensor(buf67, (1, 128, 512), (65536, 512, 1), 0); del buf67  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_21(c_void_p(buf115.data_ptr()), c_void_p(add_127.data_ptr()), c_void_p(rsqrt_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del add_127
    del getitem_145
    del primals_37
    del rsqrt_36
    buf117 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (512, 128), (1, 512), 0), view_504, out=buf117)
    del view_504
    buf118 = buf110; del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (128, 512), (512, 1), 0), permute_372, out=buf118)
    del permute_372
    buf119 = buf103; del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_375, reinterpret_tensor(buf118, (6, 128, 64), (64, 384, 1), 0), out=buf119)
    del permute_375
    buf120 = buf100; del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf118, (6, 128, 64), (64, 384, 1), 0), permute_376, out=buf120)
    del permute_376
    buf121 = buf95; del buf95  # reuse
    buf122 = buf120; del buf120  # reuse
    buf123 = buf97; del buf97  # reuse
    buf126 = buf96; del buf96  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_22(c_void_p(buf122.data_ptr()), c_void_p(getitem_143.data_ptr()), c_void_p(alias_95.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()))
    del alias_95
    del getitem_143
    buf128 = reinterpret_tensor(buf118, (6, 64, 128), (8192, 128, 1), 0); del buf118  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_377, buf126, out=buf128)
    del permute_377
    buf129 = reinterpret_tensor(buf102, (6, 128, 64), (8192, 64, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf126, permute_378, out=buf129)
    del permute_378
    buf130 = reinterpret_tensor(buf93, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf93  # reuse
    cpp_fused_clone_23(c_void_p(tangents_28.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf130.data_ptr()))
    del tangents_28
    buf131 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (384, 128), (1, 384), 0), view_486, out=buf131)
    buf132 = reinterpret_tensor(buf116, (128, 512), (512, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (128, 384), (384, 1), 0), permute_383, out=buf132)
    del permute_383
    buf133 = buf130; del buf130  # reuse
    cpp_fused_clone_24(c_void_p(tangents_27.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf133.data_ptr()))
    del tangents_27
    buf134 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (384, 128), (1, 384), 0), view_486, out=buf134)
    buf135 = reinterpret_tensor(buf89, (128, 512), (512, 1), 0); del buf89  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (128, 384), (384, 1), 0), permute_388, out=buf135)
    del permute_388
    buf136 = reinterpret_tensor(buf133, (128, 384), (384, 1), 0); del buf133  # reuse
    cpp_fused_view_25(c_void_p(buf129.data_ptr()), c_void_p(buf136.data_ptr()))
    buf137 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (384, 128), (1, 384), 0), view_486, out=buf137)
    del view_486
    buf138 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf136, permute_393, out=buf138)
    del permute_393
    buf139 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf140 = buf114; del buf114  # reuse
    buf141 = buf115; del buf115  # reuse
    buf142 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_26(c_void_p(buf141.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(add_124.data_ptr()), c_void_p(rsqrt_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(getitem_141.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()))
    del add_124
    del getitem_141
    del primals_36
    del rsqrt_35
    buf143 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (512, 128), (1, 512), 0), view_484, out=buf143)
    del view_484
    buf144 = reinterpret_tensor(buf84, (128, 1024), (1024, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (128, 512), (512, 1), 0), permute_397, out=buf144)
    del permute_397
    buf145 = buf80; del buf80  # reuse
    buf148 = reinterpret_tensor(buf79, (1, 128, 1024), (131072, 1024, 1), 0); del buf79  # reuse
    buf149 = buf148; del buf148  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_27(c_void_p(buf149.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(getitem_139.data_ptr()), c_void_p(mm_119.data_ptr()), c_void_p(tanh_13.data_ptr()), c_void_p(mm_120.data_ptr()), c_void_p(buf145.data_ptr()))
    del getitem_139
    del mm_119
    del mm_120
    del tanh_13
    buf146 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (1024, 128), (1, 1024), 0), view_480, out=buf146)
    buf147 = reinterpret_tensor(buf142, (128, 512), (512, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (128, 1024), (1024, 1), 0), permute_401, out=buf147)
    del permute_401
    buf150 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (1024, 128), (1, 1024), 0), view_480, out=buf150)
    del view_480
    buf151 = buf138; del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0), permute_405, out=buf151)
    del permute_405
    buf152 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf153 = buf140; del buf140  # reuse
    buf154 = buf141; del buf141  # reuse
    buf155 = reinterpret_tensor(buf135, (1, 128, 512), (65536, 512, 1), 0); del buf135  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_28(c_void_p(buf154.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(add_120.data_ptr()), c_void_p(rsqrt_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(getitem_137.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()))
    del add_120
    del getitem_137
    del primals_35
    del rsqrt_34
    buf156 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (512, 128), (1, 512), 0), view_478, out=buf156)
    del view_478
    buf157 = buf136; del buf136  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (128, 512), (512, 1), 0), permute_409, out=buf157)
    del permute_409
    buf158 = buf129; del buf129  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_412, reinterpret_tensor(buf157, (6, 128, 64), (64, 384, 1), 0), out=buf158)
    del permute_412
    buf159 = buf126; del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf157, (6, 128, 64), (64, 384, 1), 0), permute_413, out=buf159)
    del permute_413
    buf160 = buf121; del buf121  # reuse
    buf161 = buf159; del buf159  # reuse
    buf162 = reinterpret_tensor(buf122, (98304, ), (1, ), 0); del buf122  # reuse
    buf165 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_29(c_void_p(buf161.data_ptr()), c_void_p(getitem_135.data_ptr()), c_void_p(alias_99.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf165.data_ptr()))
    del alias_99
    del getitem_135
    buf167 = reinterpret_tensor(buf157, (6, 64, 128), (8192, 128, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_414, buf165, out=buf167)
    del permute_414
    buf168 = reinterpret_tensor(buf128, (6, 128, 64), (8192, 64, 1), 0); del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf165, permute_415, out=buf168)
    del permute_415
    buf169 = reinterpret_tensor(buf119, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf119  # reuse
    cpp_fused_clone_30(c_void_p(tangents_26.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf169.data_ptr()))
    del tangents_26
    buf170 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (384, 128), (1, 384), 0), view_233, out=buf170)
    buf171 = reinterpret_tensor(buf155, (128, 512), (512, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (128, 384), (384, 1), 0), permute_420, out=buf171)
    del permute_420
    buf172 = buf169; del buf169  # reuse
    cpp_fused_clone_31(c_void_p(tangents_25.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf172.data_ptr()))
    del tangents_25
    buf173 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (384, 128), (1, 384), 0), view_233, out=buf173)
    buf174 = buf151; del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (128, 384), (384, 1), 0), permute_425, out=buf174)
    del permute_425
    buf175 = reinterpret_tensor(buf172, (128, 384), (384, 1), 0); del buf172  # reuse
    cpp_fused_view_32(c_void_p(buf168.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (384, 128), (1, 384), 0), view_460, out=buf176)
    del view_460
    buf177 = buf147; del buf147  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf175, permute_430, out=buf177)
    del permute_430
    buf178 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf179 = buf153; del buf153  # reuse
    buf180 = buf154; del buf154  # reuse
    buf181 = reinterpret_tensor(buf132, (1, 128, 512), (65536, 512, 1), 0); del buf132  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_33(c_void_p(buf180.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(add_117.data_ptr()), c_void_p(rsqrt_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(getitem_133.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()))
    del add_117
    del getitem_133
    del primals_34
    del rsqrt_33
    buf182 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (512, 128), (1, 512), 0), view_458, out=buf182)
    del view_458
    buf183 = buf175; del buf175  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (128, 512), (512, 1), 0), permute_434, out=buf183)
    del permute_434
    buf184 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_437, reinterpret_tensor(buf183, (6, 128, 64), (64, 384, 1), 0), out=buf184)
    del permute_437
    buf185 = buf165; del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf183, (6, 128, 64), (64, 384, 1), 0), permute_438, out=buf185)
    del permute_438
    buf186 = buf160; del buf160  # reuse
    buf187 = buf185; del buf185  # reuse
    buf188 = buf162; del buf162  # reuse
    buf191 = buf161; del buf161  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_34(c_void_p(buf187.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(alias_101.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf191.data_ptr()))
    del alias_101
    del getitem_131
    buf193 = reinterpret_tensor(buf183, (6, 64, 128), (8192, 128, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_439, buf191, out=buf193)
    del permute_439
    buf194 = reinterpret_tensor(buf167, (6, 128, 64), (8192, 64, 1), 0); del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf191, permute_440, out=buf194)
    del permute_440
    buf195 = reinterpret_tensor(buf158, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf158  # reuse
    cpp_fused_clone_35(c_void_p(tangents_24.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf195.data_ptr()))
    del tangents_24
    buf196 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (384, 128), (1, 384), 0), view_440, out=buf196)
    buf197 = reinterpret_tensor(buf181, (128, 512), (512, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (128, 384), (384, 1), 0), permute_445, out=buf197)
    del permute_445
    buf198 = buf195; del buf195  # reuse
    cpp_fused_clone_36(c_void_p(tangents_23.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf198.data_ptr()))
    del tangents_23
    buf199 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (384, 128), (1, 384), 0), view_440, out=buf199)
    buf200 = buf177; del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (128, 384), (384, 1), 0), permute_450, out=buf200)
    del permute_450
    buf201 = reinterpret_tensor(buf198, (128, 384), (384, 1), 0); del buf198  # reuse
    cpp_fused_view_37(c_void_p(buf194.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (384, 128), (1, 384), 0), view_440, out=buf202)
    del view_440
    buf203 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf201, permute_455, out=buf203)
    del permute_455
    buf204 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf205 = buf179; del buf179  # reuse
    buf206 = buf180; del buf180  # reuse
    buf207 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_38(c_void_p(buf206.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(add_114.data_ptr()), c_void_p(rsqrt_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(getitem_129.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del add_114
    del getitem_129
    del primals_33
    del rsqrt_32
    buf208 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (512, 128), (1, 512), 0), view_438, out=buf208)
    del view_438
    buf209 = reinterpret_tensor(buf149, (128, 1024), (1024, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (128, 512), (512, 1), 0), permute_459, out=buf209)
    del permute_459
    buf210 = buf145; del buf145  # reuse
    buf213 = reinterpret_tensor(buf144, (1, 128, 1024), (131072, 1024, 1), 0); del buf144  # reuse
    buf214 = buf213; del buf213  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_39(c_void_p(buf214.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(mm_108.data_ptr()), c_void_p(tanh_12.data_ptr()), c_void_p(mm_109.data_ptr()), c_void_p(buf210.data_ptr()))
    del getitem_127
    del mm_108
    del mm_109
    del tanh_12
    buf211 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (1024, 128), (1, 1024), 0), view_434, out=buf211)
    buf212 = reinterpret_tensor(buf207, (128, 512), (512, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (128, 1024), (1024, 1), 0), permute_463, out=buf212)
    del permute_463
    buf215 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf214, (1024, 128), (1, 1024), 0), view_434, out=buf215)
    del view_434
    buf216 = buf203; del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf214, (128, 1024), (1024, 1), 0), permute_467, out=buf216)
    del permute_467
    buf217 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf218 = buf205; del buf205  # reuse
    buf219 = buf206; del buf206  # reuse
    buf220 = reinterpret_tensor(buf200, (1, 128, 512), (65536, 512, 1), 0); del buf200  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_40(c_void_p(buf219.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(add_110.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(getitem_125.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()))
    del add_110
    del getitem_125
    del primals_32
    del rsqrt_31
    buf221 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (512, 128), (1, 512), 0), view_432, out=buf221)
    del view_432
    buf222 = buf201; del buf201  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (128, 512), (512, 1), 0), permute_471, out=buf222)
    del permute_471
    buf223 = buf194; del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_474, reinterpret_tensor(buf222, (6, 128, 64), (64, 384, 1), 0), out=buf223)
    del permute_474
    buf224 = buf191; del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf222, (6, 128, 64), (64, 384, 1), 0), permute_475, out=buf224)
    del permute_475
    buf225 = buf186; del buf186  # reuse
    buf226 = buf224; del buf224  # reuse
    buf227 = reinterpret_tensor(buf187, (98304, ), (1, ), 0); del buf187  # reuse
    buf230 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_41(c_void_p(buf226.data_ptr()), c_void_p(getitem_123.data_ptr()), c_void_p(alias_105.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf230.data_ptr()))
    del alias_105
    del getitem_123
    buf232 = reinterpret_tensor(buf222, (6, 64, 128), (8192, 128, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_476, buf230, out=buf232)
    del permute_476
    buf233 = reinterpret_tensor(buf193, (6, 128, 64), (8192, 64, 1), 0); del buf193  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf230, permute_477, out=buf233)
    del permute_477
    buf234 = reinterpret_tensor(buf184, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf184  # reuse
    cpp_fused_clone_42(c_void_p(tangents_22.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf234.data_ptr()))
    del tangents_22
    buf235 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (384, 128), (1, 384), 0), view_233, out=buf235)
    buf236 = reinterpret_tensor(buf220, (128, 512), (512, 1), 0); del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (128, 384), (384, 1), 0), permute_482, out=buf236)
    del permute_482
    buf237 = buf234; del buf234  # reuse
    cpp_fused_clone_43(c_void_p(tangents_21.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf237.data_ptr()))
    del tangents_21
    buf238 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (384, 128), (1, 384), 0), view_233, out=buf238)
    buf239 = buf216; del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (128, 384), (384, 1), 0), permute_487, out=buf239)
    del permute_487
    buf241 = reinterpret_tensor(buf237, (128, 384), (384, 1), 0); del buf237  # reuse
    cpp_fused_view_44(c_void_p(buf233.data_ptr()), c_void_p(buf241.data_ptr()))
    buf243 = buf212; del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf241, permute_492, out=buf243)
    del permute_492
    buf245 = buf218; del buf218  # reuse
    buf246 = buf219; del buf219  # reuse
    buf247 = reinterpret_tensor(buf197, (1, 128, 512), (65536, 512, 1), 0); del buf197  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_45(c_void_p(buf246.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(add_107.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del getitem_121
    del primals_31
    buf249 = reinterpret_tensor(buf233, (128, 384), (384, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (128, 512), (512, 1), 0), permute_496, out=buf249)
    del permute_496
    buf250 = reinterpret_tensor(buf232, (6, 128, 64), (8192, 64, 1), 0); del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_499, reinterpret_tensor(buf249, (6, 128, 64), (64, 384, 1), 0), out=buf250)
    del permute_499
    buf261 = reinterpret_tensor(buf223, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf223  # reuse
    cpp_fused_clone_46(c_void_p(tangents_20.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf261.data_ptr()))
    del tangents_20
    buf263 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (128, 384), (384, 1), 0), permute_507, out=buf263)
    del permute_507
    buf251 = buf230; del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf249, (6, 128, 64), (64, 384, 1), 0), permute_500, out=buf251)
    del permute_500
    buf252 = buf225; del buf225  # reuse
    buf253 = buf251; del buf251  # reuse
    buf254 = buf227; del buf227  # reuse
    buf257 = buf226; del buf226  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_47(c_void_p(buf253.data_ptr()), c_void_p(getitem_119.data_ptr()), c_void_p(alias_107.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()))
    del alias_107
    del getitem_119
    buf259 = reinterpret_tensor(buf249, (6, 64, 128), (8192, 128, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_501, buf257, out=buf259)
    del permute_501
    buf264 = reinterpret_tensor(buf250, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf250  # reuse
    cpp_fused_clone_48(c_void_p(tangents_19.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf264.data_ptr()))
    del tangents_19
    buf266 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (128, 384), (384, 1), 0), permute_512, out=buf266)
    del permute_512
    buf260 = reinterpret_tensor(buf259, (6, 128, 64), (8192, 64, 1), 0); del buf259  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf257, permute_502, out=buf260)
    del permute_502
    buf267 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_49(c_void_p(buf260.data_ptr()), c_void_p(buf267.data_ptr()))
    buf269 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf267, permute_517, out=buf269)
    del permute_517
    buf271 = buf245; del buf245  # reuse
    buf272 = buf246; del buf246  # reuse
    buf273 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_50(c_void_p(buf272.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(add_104.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    del getitem_117
    del primals_30
    buf275 = reinterpret_tensor(buf214, (128, 1024), (1024, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (128, 512), (512, 1), 0), permute_521, out=buf275)
    del permute_521
    buf276 = buf210; del buf210  # reuse
    buf279 = reinterpret_tensor(buf209, (1, 128, 1024), (131072, 1024, 1), 0); del buf209  # reuse
    buf280 = buf279; del buf279  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_51(c_void_p(buf280.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(getitem_115.data_ptr()), c_void_p(mm_97.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(mm_98.data_ptr()), c_void_p(buf276.data_ptr()))
    del getitem_115
    del mm_97
    del mm_98
    del tanh_11
    buf278 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (128, 1024), (1024, 1), 0), permute_525, out=buf278)
    del permute_525
    buf282 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (128, 1024), (1024, 1), 0), permute_529, out=buf282)
    del permute_529
    buf284 = buf271; del buf271  # reuse
    buf285 = buf272; del buf272  # reuse
    buf286 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_52(c_void_p(buf285.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(add_100.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(getitem_113.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()))
    del getitem_113
    del primals_29
    buf288 = reinterpret_tensor(buf260, (128, 384), (384, 1), 0); del buf260  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (128, 512), (512, 1), 0), permute_533, out=buf288)
    del permute_533
    buf289 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_536, reinterpret_tensor(buf288, (6, 128, 64), (64, 384, 1), 0), out=buf289)
    del permute_536
    buf300 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_53(c_void_p(tangents_18.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf300.data_ptr()))
    del tangents_18
    buf302 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf300, (128, 384), (384, 1), 0), permute_544, out=buf302)
    del permute_544
    buf290 = buf257; del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf288, (6, 128, 64), (64, 384, 1), 0), permute_537, out=buf290)
    del permute_537
    buf291 = buf252; del buf252  # reuse
    buf292 = buf290; del buf290  # reuse
    buf293 = reinterpret_tensor(buf253, (98304, ), (1, ), 0); del buf253  # reuse
    buf296 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_54(c_void_p(buf292.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(alias_111.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf296.data_ptr()))
    del alias_111
    del getitem_111
    buf298 = reinterpret_tensor(buf288, (6, 64, 128), (8192, 128, 1), 0); del buf288  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_538, buf296, out=buf298)
    del permute_538
    buf303 = reinterpret_tensor(buf289, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf289  # reuse
    cpp_fused_clone_55(c_void_p(tangents_17.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf303.data_ptr()))
    del tangents_17
    buf305 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (128, 384), (384, 1), 0), permute_549, out=buf305)
    del permute_549
    buf299 = reinterpret_tensor(buf298, (6, 128, 64), (8192, 64, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf296, permute_539, out=buf299)
    del permute_539
    buf306 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_56(c_void_p(buf299.data_ptr()), c_void_p(buf306.data_ptr()))
    buf308 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf306, permute_554, out=buf308)
    del permute_554
    buf310 = buf284; del buf284  # reuse
    buf311 = buf285; del buf285  # reuse
    buf312 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_57(c_void_p(buf311.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(add_97.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(getitem_109.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()))
    del getitem_109
    del primals_28
    buf314 = reinterpret_tensor(buf299, (128, 384), (384, 1), 0); del buf299  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (128, 512), (512, 1), 0), permute_558, out=buf314)
    del permute_558
    buf315 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_561, reinterpret_tensor(buf314, (6, 128, 64), (64, 384, 1), 0), out=buf315)
    del permute_561
    buf326 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_58(c_void_p(tangents_16.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf326.data_ptr()))
    del tangents_16
    buf328 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (128, 384), (384, 1), 0), permute_569, out=buf328)
    del permute_569
    buf316 = buf296; del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf314, (6, 128, 64), (64, 384, 1), 0), permute_562, out=buf316)
    del permute_562
    buf317 = buf291; del buf291  # reuse
    buf318 = buf316; del buf316  # reuse
    buf319 = buf293; del buf293  # reuse
    buf322 = buf292; del buf292  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_59(c_void_p(buf318.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(alias_113.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf322.data_ptr()))
    del alias_113
    del getitem_107
    buf324 = reinterpret_tensor(buf314, (6, 64, 128), (8192, 128, 1), 0); del buf314  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_563, buf322, out=buf324)
    del permute_563
    buf329 = reinterpret_tensor(buf315, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf315  # reuse
    cpp_fused_clone_60(c_void_p(tangents_15.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf329.data_ptr()))
    del tangents_15
    buf331 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf329, (128, 384), (384, 1), 0), permute_574, out=buf331)
    del permute_574
    buf325 = reinterpret_tensor(buf324, (6, 128, 64), (8192, 64, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf322, permute_564, out=buf325)
    del permute_564
    buf332 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_61(c_void_p(buf325.data_ptr()), c_void_p(buf332.data_ptr()))
    buf334 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf332, permute_579, out=buf334)
    del permute_579
    buf336 = buf310; del buf310  # reuse
    buf337 = buf311; del buf311  # reuse
    buf338 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_62(c_void_p(buf337.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(add_94.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(getitem_105.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()))
    del getitem_105
    del primals_27
    buf340 = buf275; del buf275  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (128, 512), (512, 1), 0), permute_583, out=buf340)
    del permute_583
    buf341 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf344 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf345 = buf344; del buf344  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_63(c_void_p(buf345.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(getitem_103.data_ptr()), c_void_p(mm_86.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(mm_87.data_ptr()), c_void_p(buf341.data_ptr()))
    del getitem_103
    del mm_86
    del mm_87
    del tanh_10
    buf343 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (128, 1024), (1024, 1), 0), permute_587, out=buf343)
    del permute_587
    buf347 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (128, 1024), (1024, 1), 0), permute_591, out=buf347)
    del permute_591
    buf349 = buf336; del buf336  # reuse
    buf350 = buf337; del buf337  # reuse
    buf351 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_64(c_void_p(buf350.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(add_90.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del getitem_101
    del primals_26
    buf353 = reinterpret_tensor(buf325, (128, 384), (384, 1), 0); del buf325  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (128, 512), (512, 1), 0), permute_595, out=buf353)
    del permute_595
    buf354 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_598, reinterpret_tensor(buf353, (6, 128, 64), (64, 384, 1), 0), out=buf354)
    del permute_598
    buf365 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_65(c_void_p(tangents_14.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf365.data_ptr()))
    del tangents_14
    buf367 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (128, 384), (384, 1), 0), permute_606, out=buf367)
    del permute_606
    buf355 = buf322; del buf322  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (6, 128, 64), (64, 384, 1), 0), permute_599, out=buf355)
    del permute_599
    buf356 = buf317; del buf317  # reuse
    buf357 = buf355; del buf355  # reuse
    buf358 = reinterpret_tensor(buf318, (98304, ), (1, ), 0); del buf318  # reuse
    buf361 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_66(c_void_p(buf357.data_ptr()), c_void_p(getitem_99.data_ptr()), c_void_p(alias_117.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf361.data_ptr()))
    del alias_117
    del getitem_99
    buf363 = reinterpret_tensor(buf353, (6, 64, 128), (8192, 128, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_600, buf361, out=buf363)
    del permute_600
    buf368 = reinterpret_tensor(buf354, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf354  # reuse
    cpp_fused_clone_67(c_void_p(tangents_13.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf368.data_ptr()))
    del tangents_13
    buf370 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (128, 384), (384, 1), 0), permute_611, out=buf370)
    del permute_611
    buf364 = reinterpret_tensor(buf363, (6, 128, 64), (8192, 64, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf361, permute_601, out=buf364)
    del permute_601
    buf371 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_68(c_void_p(buf364.data_ptr()), c_void_p(buf371.data_ptr()))
    buf373 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf371, permute_616, out=buf373)
    del permute_616
    buf375 = buf349; del buf349  # reuse
    buf376 = buf350; del buf350  # reuse
    buf377 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_69(c_void_p(buf376.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(add_87.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()))
    del getitem_97
    del primals_25
    buf379 = reinterpret_tensor(buf364, (128, 384), (384, 1), 0); del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (128, 512), (512, 1), 0), permute_620, out=buf379)
    del permute_620
    buf380 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_623, reinterpret_tensor(buf379, (6, 128, 64), (64, 384, 1), 0), out=buf380)
    del permute_623
    buf391 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(tangents_12.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf391.data_ptr()))
    del tangents_12
    buf393 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (128, 384), (384, 1), 0), permute_631, out=buf393)
    del permute_631
    buf381 = buf361; del buf361  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (6, 128, 64), (64, 384, 1), 0), permute_624, out=buf381)
    del permute_624
    buf382 = buf356; del buf356  # reuse
    buf383 = buf381; del buf381  # reuse
    buf384 = buf358; del buf358  # reuse
    buf387 = buf357; del buf357  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_71(c_void_p(buf383.data_ptr()), c_void_p(getitem_95.data_ptr()), c_void_p(alias_119.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf387.data_ptr()))
    del alias_119
    del getitem_95
    buf389 = reinterpret_tensor(buf379, (6, 64, 128), (8192, 128, 1), 0); del buf379  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_625, buf387, out=buf389)
    del permute_625
    buf394 = reinterpret_tensor(buf380, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf380  # reuse
    cpp_fused_clone_72(c_void_p(tangents_11.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf394.data_ptr()))
    del tangents_11
    buf396 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (128, 384), (384, 1), 0), permute_636, out=buf396)
    del permute_636
    buf390 = reinterpret_tensor(buf389, (6, 128, 64), (8192, 64, 1), 0); del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf387, permute_626, out=buf390)
    del permute_626
    buf397 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_73(c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()))
    buf399 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf397, permute_641, out=buf399)
    del permute_641
    buf401 = buf375; del buf375  # reuse
    buf402 = buf376; del buf376  # reuse
    buf403 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_74(c_void_p(buf402.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(add_84.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(getitem_93.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()))
    del getitem_93
    del primals_24
    buf405 = buf340; del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (128, 512), (512, 1), 0), permute_645, out=buf405)
    del permute_645
    buf406 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf409 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf410 = buf409; del buf409  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_75(c_void_p(buf410.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(mm_75.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(mm_76.data_ptr()), c_void_p(buf406.data_ptr()))
    del getitem_91
    del mm_75
    del mm_76
    del tanh_9
    buf408 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (128, 1024), (1024, 1), 0), permute_649, out=buf408)
    del permute_649
    buf412 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf410, (128, 1024), (1024, 1), 0), permute_653, out=buf412)
    del permute_653
    buf414 = buf401; del buf401  # reuse
    buf415 = buf402; del buf402  # reuse
    buf416 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_76(c_void_p(buf415.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(add_80.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(getitem_89.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf416.data_ptr()))
    del getitem_89
    del primals_23
    buf418 = reinterpret_tensor(buf390, (128, 384), (384, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (128, 512), (512, 1), 0), permute_657, out=buf418)
    del permute_657
    buf419 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_660, reinterpret_tensor(buf418, (6, 128, 64), (64, 384, 1), 0), out=buf419)
    del permute_660
    buf430 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_77(c_void_p(tangents_10.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf430.data_ptr()))
    del tangents_10
    buf432 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (128, 384), (384, 1), 0), permute_668, out=buf432)
    del permute_668
    buf420 = buf387; del buf387  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (6, 128, 64), (64, 384, 1), 0), permute_661, out=buf420)
    del permute_661
    buf421 = buf382; del buf382  # reuse
    buf422 = buf420; del buf420  # reuse
    buf423 = reinterpret_tensor(buf383, (98304, ), (1, ), 0); del buf383  # reuse
    buf426 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_78(c_void_p(buf422.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(alias_123.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf426.data_ptr()))
    del alias_123
    del getitem_87
    buf428 = reinterpret_tensor(buf418, (6, 64, 128), (8192, 128, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_662, buf426, out=buf428)
    del permute_662
    buf433 = reinterpret_tensor(buf419, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf419  # reuse
    cpp_fused_clone_79(c_void_p(tangents_9.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf433.data_ptr()))
    del tangents_9
    buf435 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (128, 384), (384, 1), 0), permute_673, out=buf435)
    del permute_673
    buf429 = reinterpret_tensor(buf428, (6, 128, 64), (8192, 64, 1), 0); del buf428  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf426, permute_663, out=buf429)
    del permute_663
    buf436 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_80(c_void_p(buf429.data_ptr()), c_void_p(buf436.data_ptr()))
    buf438 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf436, permute_678, out=buf438)
    del permute_678
    buf440 = buf414; del buf414  # reuse
    buf441 = buf415; del buf415  # reuse
    buf442 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_81(c_void_p(buf441.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(add_77.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(getitem_85.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()))
    del getitem_85
    del primals_22
    buf444 = reinterpret_tensor(buf429, (128, 384), (384, 1), 0); del buf429  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (128, 512), (512, 1), 0), permute_682, out=buf444)
    del permute_682
    buf445 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_685, reinterpret_tensor(buf444, (6, 128, 64), (64, 384, 1), 0), out=buf445)
    del permute_685
    buf456 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(tangents_8.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf456.data_ptr()))
    del tangents_8
    buf458 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf456, (128, 384), (384, 1), 0), permute_693, out=buf458)
    del permute_693
    buf446 = buf426; del buf426  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf444, (6, 128, 64), (64, 384, 1), 0), permute_686, out=buf446)
    del permute_686
    buf447 = buf421; del buf421  # reuse
    buf448 = buf446; del buf446  # reuse
    buf449 = buf423; del buf423  # reuse
    buf452 = buf422; del buf422  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_83(c_void_p(buf448.data_ptr()), c_void_p(getitem_83.data_ptr()), c_void_p(alias_125.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf452.data_ptr()))
    del alias_125
    del getitem_83
    buf454 = reinterpret_tensor(buf444, (6, 64, 128), (8192, 128, 1), 0); del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_687, buf452, out=buf454)
    del permute_687
    buf459 = reinterpret_tensor(buf445, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf445  # reuse
    cpp_fused_clone_84(c_void_p(tangents_7.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf459.data_ptr()))
    del tangents_7
    buf461 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (128, 384), (384, 1), 0), permute_698, out=buf461)
    del permute_698
    buf455 = reinterpret_tensor(buf454, (6, 128, 64), (8192, 64, 1), 0); del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf452, permute_688, out=buf455)
    del permute_688
    buf462 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_85(c_void_p(buf455.data_ptr()), c_void_p(buf462.data_ptr()))
    buf464 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf462, permute_703, out=buf464)
    del permute_703
    buf466 = buf440; del buf440  # reuse
    buf467 = buf441; del buf441  # reuse
    buf468 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_86(c_void_p(buf467.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(add_74.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()))
    del getitem_81
    del primals_21
    buf470 = buf405; del buf405  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (128, 512), (512, 1), 0), permute_707, out=buf470)
    del permute_707
    buf471 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf474 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf475 = buf474; del buf474  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_87(c_void_p(buf475.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(mm_64.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(buf471.data_ptr()))
    del buf470
    del getitem_79
    del mm_64
    del mm_65
    del tanh_8
    buf473 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf471, (128, 1024), (1024, 1), 0), permute_711, out=buf473)
    del permute_711
    buf477 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf475, (128, 1024), (1024, 1), 0), permute_715, out=buf477)
    del permute_715
    buf479 = buf466; del buf466  # reuse
    buf480 = buf467; del buf467  # reuse
    buf481 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_88(c_void_p(buf480.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf481.data_ptr()))
    del getitem_77
    del primals_20
    buf483 = reinterpret_tensor(buf455, (128, 384), (384, 1), 0); del buf455  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf481, (128, 512), (512, 1), 0), permute_719, out=buf483)
    del permute_719
    buf484 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_722, reinterpret_tensor(buf483, (6, 128, 64), (64, 384, 1), 0), out=buf484)
    del permute_722
    buf495 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_89(c_void_p(tangents_6.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf495.data_ptr()))
    del tangents_6
    buf497 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (128, 384), (384, 1), 0), permute_730, out=buf497)
    del permute_730
    buf485 = buf452; del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf483, (6, 128, 64), (64, 384, 1), 0), permute_723, out=buf485)
    del permute_723
    buf486 = buf447; del buf447  # reuse
    buf487 = buf485; del buf485  # reuse
    buf488 = reinterpret_tensor(buf448, (98304, ), (1, ), 0); del buf448  # reuse
    buf491 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_90(c_void_p(buf487.data_ptr()), c_void_p(getitem_75.data_ptr()), c_void_p(alias_129.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf491.data_ptr()))
    del alias_129
    del getitem_75
    buf493 = reinterpret_tensor(buf483, (6, 64, 128), (8192, 128, 1), 0); del buf483  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_724, buf491, out=buf493)
    del permute_724
    buf498 = reinterpret_tensor(buf484, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf484  # reuse
    cpp_fused_clone_91(c_void_p(tangents_5.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf498.data_ptr()))
    del buf493
    del tangents_5
    buf500 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf498, (128, 384), (384, 1), 0), permute_735, out=buf500)
    del permute_735
    buf240 = reinterpret_tensor(buf106, (1, 128, 512), (65536, 512, 1), 0); del buf106  # reuse
    buf501 = buf240; del buf240  # reuse
    cpp_fused_add_92(c_void_p(buf501.data_ptr()), c_void_p(tangents_35.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf500.data_ptr()))
    del buf109
    del buf171
    del buf174
    del buf236
    del buf239
    del buf302
    del buf305
    del buf367
    del buf370
    del buf41
    del buf432
    del buf435
    del buf44
    del buf497
    del buf500
    del tangents_35
    buf242 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf241, (384, 128), (1, 384), 0), view_414, out=buf242)
    del buf241
    del view_414
    buf244 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_93(c_void_p(buf243.data_ptr()), c_void_p(add_107.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf244.data_ptr()))
    del add_107
    del buf243
    del rsqrt_30
    buf248 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (512, 128), (1, 512), 0), view_412, out=buf248)
    del buf247
    del view_412
    buf262 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (384, 128), (1, 384), 0), view_394, out=buf262)
    del buf261
    buf265 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (384, 128), (1, 384), 0), view_394, out=buf265)
    del buf264
    buf268 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (384, 128), (1, 384), 0), view_394, out=buf268)
    del buf267
    del view_394
    buf270 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_94(c_void_p(buf263.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(add_104.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(buf270.data_ptr()))
    del add_104
    del buf263
    del buf266
    del buf269
    del rsqrt_29
    buf274 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (512, 128), (1, 512), 0), view_392, out=buf274)
    del buf273
    del view_392
    buf277 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (1024, 128), (1, 1024), 0), view_388, out=buf277)
    del buf276
    buf281 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (1024, 128), (1, 1024), 0), view_388, out=buf281)
    del buf280
    del view_388
    buf283 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_95(c_void_p(buf278.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(add_100.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(buf283.data_ptr()))
    del add_100
    del buf278
    del buf282
    del rsqrt_28
    buf287 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (512, 128), (1, 512), 0), view_386, out=buf287)
    del buf286
    del view_386
    buf301 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf300, (384, 128), (1, 384), 0), view_233, out=buf301)
    del buf300
    buf304 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (384, 128), (1, 384), 0), view_233, out=buf304)
    del buf303
    buf307 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (384, 128), (1, 384), 0), view_368, out=buf307)
    del buf306
    del view_368
    buf309 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_96(c_void_p(buf308.data_ptr()), c_void_p(add_97.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(buf309.data_ptr()))
    del add_97
    del buf308
    del rsqrt_27
    buf313 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (512, 128), (1, 512), 0), view_366, out=buf313)
    del buf312
    del view_366
    buf327 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (384, 128), (1, 384), 0), view_348, out=buf327)
    del buf326
    buf330 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf329, (384, 128), (1, 384), 0), view_348, out=buf330)
    del buf329
    buf333 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf332, (384, 128), (1, 384), 0), view_348, out=buf333)
    del buf332
    del view_348
    buf335 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_97(c_void_p(buf328.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(add_94.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf335.data_ptr()))
    del add_94
    del buf328
    del buf331
    del buf334
    del rsqrt_26
    buf339 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (512, 128), (1, 512), 0), view_346, out=buf339)
    del buf338
    del view_346
    buf342 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (1024, 128), (1, 1024), 0), view_342, out=buf342)
    del buf341
    buf346 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (1024, 128), (1, 1024), 0), view_342, out=buf346)
    del buf345
    del view_342
    buf348 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_98(c_void_p(buf343.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(add_90.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf348.data_ptr()))
    del add_90
    del buf343
    del buf347
    del rsqrt_25
    buf352 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (512, 128), (1, 512), 0), view_340, out=buf352)
    del buf351
    del view_340
    buf366 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (384, 128), (1, 384), 0), view_233, out=buf366)
    del buf365
    buf369 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (384, 128), (1, 384), 0), view_233, out=buf369)
    del buf368
    buf372 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (384, 128), (1, 384), 0), view_322, out=buf372)
    del buf371
    del view_322
    buf374 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_99(c_void_p(buf373.data_ptr()), c_void_p(add_87.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf374.data_ptr()))
    del add_87
    del buf373
    del rsqrt_24
    buf378 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (512, 128), (1, 512), 0), view_320, out=buf378)
    del buf377
    del view_320
    buf392 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (384, 128), (1, 384), 0), view_302, out=buf392)
    del buf391
    buf395 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (384, 128), (1, 384), 0), view_302, out=buf395)
    del buf394
    buf398 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf397, (384, 128), (1, 384), 0), view_302, out=buf398)
    del buf397
    del view_302
    buf400 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_100(c_void_p(buf393.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(add_84.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(buf400.data_ptr()))
    del add_84
    del buf393
    del buf396
    del buf399
    del rsqrt_23
    buf404 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (512, 128), (1, 512), 0), view_300, out=buf404)
    del buf403
    del view_300
    buf407 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (1024, 128), (1, 1024), 0), view_296, out=buf407)
    del buf406
    buf411 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf410, (1024, 128), (1, 1024), 0), view_296, out=buf411)
    del view_296
    buf413 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_101(c_void_p(buf408.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(add_80.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(buf413.data_ptr()))
    del add_80
    del buf408
    del buf412
    del rsqrt_22
    buf417 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (512, 128), (1, 512), 0), view_294, out=buf417)
    del buf416
    del view_294
    buf431 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (384, 128), (1, 384), 0), view_233, out=buf431)
    del buf430
    buf434 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (384, 128), (1, 384), 0), view_233, out=buf434)
    del buf433
    buf437 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (384, 128), (1, 384), 0), view_276, out=buf437)
    del buf436
    del view_276
    buf439 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_sum_102(c_void_p(buf438.data_ptr()), c_void_p(add_77.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(buf439.data_ptr()))
    del add_77
    del buf438
    del rsqrt_21
    buf443 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (512, 128), (1, 512), 0), view_274, out=buf443)
    del buf442
    del view_274
    buf457 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf456, (384, 128), (1, 384), 0), view_256, out=buf457)
    del buf456
    buf460 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (384, 128), (1, 384), 0), view_256, out=buf460)
    buf463 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (384, 128), (1, 384), 0), view_256, out=buf463)
    del view_256
    buf465 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_103(c_void_p(buf458.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(add_74.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf465.data_ptr()))
    del add_74
    del buf458
    del buf461
    del buf464
    del rsqrt_20
    buf469 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (512, 128), (1, 512), 0), view_254, out=buf469)
    del buf468
    del view_254
    buf472 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf471, (1024, 128), (1, 1024), 0), view_250, out=buf472)
    buf476 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf475, (1024, 128), (1, 1024), 0), view_250, out=buf476)
    del view_250
    buf478 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_104(c_void_p(buf473.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(add_70.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf478.data_ptr()))
    del add_70
    del rsqrt_19
    buf482 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf481, (512, 128), (1, 512), 0), view_248, out=buf482)
    del view_248
    buf494 = reinterpret_tensor(buf462, (6, 128, 64), (8192, 64, 1), 0); del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf491, permute_725, out=buf494)
    del permute_725
    buf496 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (384, 128), (1, 384), 0), view_233, out=buf496)
    buf499 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf498, (384, 128), (1, 384), 0), view_233, out=buf499)
    del view_233
    buf502 = reinterpret_tensor(buf498, (128, 384), (384, 1), 0); del buf498  # reuse
    cpp_fused_view_105(c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()))
    buf503 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf502, (384, 128), (1, 384), 0), view_230, out=buf503)
    del view_230
    buf504 = reinterpret_tensor(buf481, (128, 512), (512, 1), 0); del buf481  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf502, permute_740, out=buf504)
    del permute_740
    buf505 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf506 = buf479; del buf479  # reuse
    buf507 = buf480; del buf480  # reuse
    buf508 = reinterpret_tensor(buf477, (1, 128, 512), (65536, 512, 1), 0); del buf477  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_106(c_void_p(buf507.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(add_66.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf508.data_ptr()))
    del add_66
    del getitem_73
    del primals_19
    del rsqrt_18
    buf509 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf508, (512, 128), (1, 512), 0), view_228, out=buf509)
    del view_228
    buf510 = buf502; del buf502  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf508, (128, 512), (512, 1), 0), permute_744, out=buf510)
    del permute_744
    buf511 = buf494; del buf494  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_747, reinterpret_tensor(buf510, (6, 128, 64), (64, 384, 1), 0), out=buf511)
    del permute_747
    buf512 = buf491; del buf491  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf510, (6, 128, 64), (64, 384, 1), 0), permute_748, out=buf512)
    del permute_748
    buf513 = buf486; del buf486  # reuse
    buf514 = buf512; del buf512  # reuse
    buf515 = buf488; del buf488  # reuse
    buf518 = buf487; del buf487  # reuse
    buf521 = empty_strided((128, 128, 6), (128, 1, 16384), device='cpu', dtype=torch.float32)
    buf520 = empty((32, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_107(c_void_p(buf514.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(alias_131.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf520.data_ptr()))
    del alias_131
    del getitem_71
    aten.index_put_(buf520, [add_63], buf521, True)
    del add_63
    buf524 = reinterpret_tensor(buf510, (6, 64, 128), (8192, 128, 1), 0); del buf510  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_750, buf518, out=buf524)
    del permute_750
    buf525 = reinterpret_tensor(buf495, (6, 128, 64), (8192, 64, 1), 0); del buf495  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf518, permute_751, out=buf525)
    del permute_751
    buf526 = buf459; del buf459  # reuse
    cpp_fused_clone_108(c_void_p(tangents_4.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf526.data_ptr()))
    del tangents_4
    buf527 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf526, (384, 128), (1, 384), 0), view_210, out=buf527)
    buf528 = reinterpret_tensor(buf508, (128, 512), (512, 1), 0); del buf508  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf526, (128, 384), (384, 1), 0), permute_756, out=buf528)
    del permute_756
    buf529 = buf526; del buf526  # reuse
    cpp_fused_clone_109(c_void_p(tangents_3.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf529.data_ptr()))
    del tangents_3
    buf530 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (384, 128), (1, 384), 0), view_210, out=buf530)
    buf531 = buf504; del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (128, 384), (384, 1), 0), permute_761, out=buf531)
    del permute_761
    buf532 = reinterpret_tensor(buf529, (128, 384), (384, 1), 0); del buf529  # reuse
    cpp_fused_view_110(c_void_p(buf525.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf532, (384, 128), (1, 384), 0), view_210, out=buf533)
    del view_210
    buf534 = buf473; del buf473  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf532, permute_766, out=buf534)
    del permute_766
    buf535 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf536 = buf506; del buf506  # reuse
    buf537 = buf507; del buf507  # reuse
    buf539 = buf537; del buf537  # reuse
    buf538 = empty((250112, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_111(c_void_p(buf539.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(getitem_68.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(view_209.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf538.data_ptr()))
    del getitem_68
    del getitem_69
    del primals_18
    del rsqrt_17
    aten.index_put_(buf538, [view_209], buf539, True)
    del view_209
    buf542 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf543 = buf536; del buf536  # reuse
    buf544 = buf501; del buf501  # reuse
    buf545 = buf539; del buf539  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_112(c_void_p(buf544.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(add_59.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(getitem_65.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf545.data_ptr()))
    del add_59
    del getitem_65
    del getitem_67
    del primals_17
    del rsqrt_16
    buf546 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (512, 128), (1, 512), 0), view_207, out=buf546)
    del view_207
    buf547 = reinterpret_tensor(buf475, (128, 1024), (1024, 1), 0); del buf475  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (128, 512), (512, 1), 0), permute_770, out=buf547)
    del permute_770
    buf548 = buf471; del buf471  # reuse
    buf551 = buf410; del buf410  # reuse
    buf552 = buf551; del buf551  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_113(c_void_p(buf552.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(getitem_63.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(mm_54.data_ptr()), c_void_p(buf548.data_ptr()))
    del getitem_63
    del mm_53
    del mm_54
    del tanh_7
    buf549 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf548, (1024, 128), (1, 1024), 0), view_203, out=buf549)
    buf550 = reinterpret_tensor(buf545, (128, 512), (512, 1), 0); del buf545  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf548, (128, 1024), (1024, 1), 0), permute_774, out=buf550)
    del permute_774
    buf553 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf552, (1024, 128), (1, 1024), 0), view_203, out=buf553)
    del view_203
    buf554 = buf534; del buf534  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf552, (128, 1024), (1024, 1), 0), permute_778, out=buf554)
    del permute_778
    buf555 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf556 = buf543; del buf543  # reuse
    buf557 = buf544; del buf544  # reuse
    buf558 = reinterpret_tensor(buf531, (1, 128, 512), (65536, 512, 1), 0); del buf531  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_114(c_void_p(buf557.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf558.data_ptr()))
    del add_55
    del getitem_61
    del primals_16
    del rsqrt_15
    buf559 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (512, 128), (1, 512), 0), view_201, out=buf559)
    del view_201
    buf560 = buf532; del buf532  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (128, 512), (512, 1), 0), permute_782, out=buf560)
    del permute_782
    buf561 = buf525; del buf525  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_785, reinterpret_tensor(buf560, (6, 128, 64), (64, 384, 1), 0), out=buf561)
    del permute_785
    buf562 = buf518; del buf518  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf560, (6, 128, 64), (64, 384, 1), 0), permute_786, out=buf562)
    del permute_786
    buf563 = buf513; del buf513  # reuse
    buf564 = buf562; del buf562  # reuse
    buf565 = reinterpret_tensor(buf521, (98304, ), (1, ), 0); del buf521  # reuse
    buf568 = reinterpret_tensor(buf58, (6, 128, 128), (16384, 128, 1), 0); del buf58  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_115(c_void_p(buf564.data_ptr()), c_void_p(getitem_59.data_ptr()), c_void_p(alias_136.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf568.data_ptr()))
    del alias_136
    del getitem_59
    buf570 = reinterpret_tensor(buf560, (6, 64, 128), (8192, 128, 1), 0); del buf560  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_787, buf568, out=buf570)
    del permute_787
    buf571 = reinterpret_tensor(buf524, (6, 128, 64), (8192, 64, 1), 0); del buf524  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf568, permute_788, out=buf571)
    del permute_788
    buf572 = reinterpret_tensor(buf511, (128, 384), (384, 1), 0); del buf511  # reuse
    cpp_fused_view_116(c_void_p(buf561.data_ptr()), c_void_p(buf572.data_ptr()))
    buf573 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf572, (384, 128), (1, 384), 0), view_183, out=buf573)
    buf574 = reinterpret_tensor(buf558, (128, 512), (512, 1), 0); del buf558  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf572, permute_793, out=buf574)
    del permute_793
    buf575 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf570, (384, 128), (128, 1), 0), view_183, out=buf575)
    buf576 = buf554; del buf554  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf570, (128, 384), (1, 128), 0), permute_798, out=buf576)
    del permute_798
    buf577 = reinterpret_tensor(buf570, (128, 384), (384, 1), 0); del buf570  # reuse
    cpp_fused_view_117(c_void_p(buf571.data_ptr()), c_void_p(buf577.data_ptr()))
    buf578 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (384, 128), (1, 384), 0), view_183, out=buf578)
    del view_183
    buf579 = buf550; del buf550  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf577, permute_803, out=buf579)
    del permute_803
    buf580 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf581 = buf556; del buf556  # reuse
    buf582 = buf557; del buf557  # reuse
    buf583 = reinterpret_tensor(buf528, (1, 128, 512), (65536, 512, 1), 0); del buf528  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_118(c_void_p(buf582.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(add_52.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()))
    del add_52
    del getitem_57
    del primals_15
    del rsqrt_14
    buf584 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf583, (512, 128), (1, 512), 0), view_181, out=buf584)
    del view_181
    buf585 = reinterpret_tensor(buf552, (128, 1024), (1024, 1), 0); del buf552  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf583, (128, 512), (512, 1), 0), permute_807, out=buf585)
    del permute_807
    buf586 = buf548; del buf548  # reuse
    buf589 = reinterpret_tensor(buf547, (1, 128, 1024), (131072, 1024, 1), 0); del buf547  # reuse
    buf590 = buf589; del buf589  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_119(c_void_p(buf590.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(mm_46.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(mm_47.data_ptr()), c_void_p(buf586.data_ptr()))
    del getitem_55
    del mm_46
    del mm_47
    del tanh_6
    buf587 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (1024, 128), (1, 1024), 0), view_177, out=buf587)
    buf588 = reinterpret_tensor(buf583, (128, 512), (512, 1), 0); del buf583  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (128, 1024), (1024, 1), 0), permute_811, out=buf588)
    del permute_811
    buf591 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf590, (1024, 128), (1, 1024), 0), view_177, out=buf591)
    del view_177
    buf592 = buf579; del buf579  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf590, (128, 1024), (1024, 1), 0), permute_815, out=buf592)
    del permute_815
    buf593 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf594 = buf581; del buf581  # reuse
    buf595 = buf582; del buf582  # reuse
    buf596 = reinterpret_tensor(buf576, (1, 128, 512), (65536, 512, 1), 0); del buf576  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_120(c_void_p(buf595.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(add_48.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf596.data_ptr()))
    del add_48
    del getitem_53
    del primals_14
    del rsqrt_13
    buf597 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf596, (512, 128), (1, 512), 0), view_175, out=buf597)
    del view_175
    buf598 = buf577; del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf596, (128, 512), (512, 1), 0), permute_819, out=buf598)
    del permute_819
    buf599 = buf571; del buf571  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_822, reinterpret_tensor(buf598, (6, 128, 64), (64, 384, 1), 0), out=buf599)
    del permute_822
    buf600 = buf568; del buf568  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf598, (6, 128, 64), (64, 384, 1), 0), permute_823, out=buf600)
    del permute_823
    buf601 = buf563; del buf563  # reuse
    buf602 = buf600; del buf600  # reuse
    buf603 = reinterpret_tensor(buf564, (98304, ), (1, ), 0); del buf564  # reuse
    buf606 = reinterpret_tensor(buf515, (6, 128, 128), (16384, 128, 1), 0); del buf515  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_121(c_void_p(buf602.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(alias_140.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf606.data_ptr()))
    del alias_140
    del getitem_51
    buf608 = reinterpret_tensor(buf598, (6, 64, 128), (8192, 128, 1), 0); del buf598  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_824, buf606, out=buf608)
    del permute_824
    buf609 = reinterpret_tensor(buf572, (6, 128, 64), (8192, 64, 1), 0); del buf572  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf606, permute_825, out=buf609)
    del permute_825
    buf610 = reinterpret_tensor(buf561, (128, 384), (384, 1), 0); del buf561  # reuse
    cpp_fused_view_122(c_void_p(buf599.data_ptr()), c_void_p(buf610.data_ptr()))
    buf611 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf610, (384, 128), (1, 384), 0), view_157, out=buf611)
    buf612 = reinterpret_tensor(buf596, (128, 512), (512, 1), 0); del buf596  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf610, permute_830, out=buf612)
    del permute_830
    buf613 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (384, 128), (128, 1), 0), view_157, out=buf613)
    buf614 = buf592; del buf592  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (128, 384), (1, 128), 0), permute_835, out=buf614)
    del permute_835
    buf615 = reinterpret_tensor(buf608, (128, 384), (384, 1), 0); del buf608  # reuse
    cpp_fused_view_123(c_void_p(buf609.data_ptr()), c_void_p(buf615.data_ptr()))
    buf616 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf615, (384, 128), (1, 384), 0), view_157, out=buf616)
    del view_157
    buf617 = buf588; del buf588  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf615, permute_840, out=buf617)
    del permute_840
    buf618 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf619 = buf594; del buf594  # reuse
    buf620 = buf595; del buf595  # reuse
    buf621 = reinterpret_tensor(buf574, (1, 128, 512), (65536, 512, 1), 0); del buf574  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_124(c_void_p(buf620.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf621.data_ptr()))
    del add_45
    del getitem_49
    del primals_13
    del rsqrt_12
    buf622 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (512, 128), (1, 512), 0), view_155, out=buf622)
    del view_155
    buf623 = reinterpret_tensor(buf590, (128, 1024), (1024, 1), 0); del buf590  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (128, 512), (512, 1), 0), permute_844, out=buf623)
    del permute_844
    buf624 = buf586; del buf586  # reuse
    buf627 = reinterpret_tensor(buf585, (1, 128, 1024), (131072, 1024, 1), 0); del buf585  # reuse
    buf628 = buf627; del buf627  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_125(c_void_p(buf628.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(mm_40.data_ptr()), c_void_p(buf624.data_ptr()))
    del getitem_47
    del mm_39
    del mm_40
    del tanh_5
    buf625 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf624, (1024, 128), (1, 1024), 0), view_151, out=buf625)
    buf626 = reinterpret_tensor(buf621, (128, 512), (512, 1), 0); del buf621  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf624, (128, 1024), (1024, 1), 0), permute_848, out=buf626)
    del permute_848
    buf629 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf628, (1024, 128), (1, 1024), 0), view_151, out=buf629)
    del view_151
    buf630 = buf617; del buf617  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf628, (128, 1024), (1024, 1), 0), permute_852, out=buf630)
    del permute_852
    buf631 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf632 = buf619; del buf619  # reuse
    buf633 = buf620; del buf620  # reuse
    buf634 = reinterpret_tensor(buf614, (1, 128, 512), (65536, 512, 1), 0); del buf614  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_126(c_void_p(buf633.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(add_41.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()))
    del add_41
    del getitem_45
    del primals_12
    del rsqrt_11
    buf635 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf634, (512, 128), (1, 512), 0), view_149, out=buf635)
    del view_149
    buf636 = buf615; del buf615  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf634, (128, 512), (512, 1), 0), permute_856, out=buf636)
    del permute_856
    buf637 = buf609; del buf609  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_859, reinterpret_tensor(buf636, (6, 128, 64), (64, 384, 1), 0), out=buf637)
    del permute_859
    buf638 = buf606; del buf606  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf636, (6, 128, 64), (64, 384, 1), 0), permute_860, out=buf638)
    del permute_860
    buf639 = buf601; del buf601  # reuse
    buf640 = buf638; del buf638  # reuse
    buf641 = reinterpret_tensor(buf602, (98304, ), (1, ), 0); del buf602  # reuse
    buf644 = buf514; del buf514  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_127(c_void_p(buf640.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(alias_144.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf644.data_ptr()))
    del alias_144
    del getitem_43
    buf646 = reinterpret_tensor(buf636, (6, 64, 128), (8192, 128, 1), 0); del buf636  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_861, buf644, out=buf646)
    del permute_861
    buf647 = reinterpret_tensor(buf610, (6, 128, 64), (8192, 64, 1), 0); del buf610  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf644, permute_862, out=buf647)
    del permute_862
    buf648 = reinterpret_tensor(buf599, (128, 384), (384, 1), 0); del buf599  # reuse
    cpp_fused_view_128(c_void_p(buf637.data_ptr()), c_void_p(buf648.data_ptr()))
    buf649 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf648, (384, 128), (1, 384), 0), view_131, out=buf649)
    buf650 = reinterpret_tensor(buf634, (128, 512), (512, 1), 0); del buf634  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf648, permute_867, out=buf650)
    del permute_867
    buf651 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf646, (384, 128), (128, 1), 0), view_131, out=buf651)
    buf652 = buf630; del buf630  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf646, (128, 384), (1, 128), 0), permute_872, out=buf652)
    del permute_872
    buf653 = reinterpret_tensor(buf646, (128, 384), (384, 1), 0); del buf646  # reuse
    cpp_fused_view_129(c_void_p(buf647.data_ptr()), c_void_p(buf653.data_ptr()))
    buf654 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf653, (384, 128), (1, 384), 0), view_131, out=buf654)
    del view_131
    buf655 = buf626; del buf626  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf653, permute_877, out=buf655)
    del permute_877
    buf656 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf657 = buf632; del buf632  # reuse
    buf658 = buf633; del buf633  # reuse
    buf659 = reinterpret_tensor(buf612, (1, 128, 512), (65536, 512, 1), 0); del buf612  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_130(c_void_p(buf658.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(add_38.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf659.data_ptr()))
    del add_38
    del getitem_41
    del primals_11
    del rsqrt_10
    buf660 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf659, (512, 128), (1, 512), 0), view_129, out=buf660)
    del view_129
    buf661 = reinterpret_tensor(buf628, (128, 1024), (1024, 1), 0); del buf628  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf659, (128, 512), (512, 1), 0), permute_881, out=buf661)
    del permute_881
    buf662 = buf624; del buf624  # reuse
    buf665 = reinterpret_tensor(buf623, (1, 128, 1024), (131072, 1024, 1), 0); del buf623  # reuse
    buf666 = buf665; del buf665  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_131(c_void_p(buf666.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(getitem_39.data_ptr()), c_void_p(mm_32.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(mm_33.data_ptr()), c_void_p(buf662.data_ptr()))
    del getitem_39
    del mm_32
    del mm_33
    del tanh_4
    buf663 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf662, (1024, 128), (1, 1024), 0), view_125, out=buf663)
    buf664 = reinterpret_tensor(buf659, (128, 512), (512, 1), 0); del buf659  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf662, (128, 1024), (1024, 1), 0), permute_885, out=buf664)
    del permute_885
    buf667 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf666, (1024, 128), (1, 1024), 0), view_125, out=buf667)
    del view_125
    buf668 = buf655; del buf655  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf666, (128, 1024), (1024, 1), 0), permute_889, out=buf668)
    del permute_889
    buf669 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf670 = buf657; del buf657  # reuse
    buf671 = buf658; del buf658  # reuse
    buf672 = reinterpret_tensor(buf652, (1, 128, 512), (65536, 512, 1), 0); del buf652  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_132(c_void_p(buf671.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(add_34.data_ptr()), c_void_p(rsqrt_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf672.data_ptr()))
    del add_34
    del getitem_37
    del primals_10
    del rsqrt_9
    buf673 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf672, (512, 128), (1, 512), 0), view_123, out=buf673)
    del view_123
    buf674 = buf653; del buf653  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf672, (128, 512), (512, 1), 0), permute_893, out=buf674)
    del permute_893
    buf675 = buf647; del buf647  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_896, reinterpret_tensor(buf674, (6, 128, 64), (64, 384, 1), 0), out=buf675)
    del permute_896
    buf676 = buf644; del buf644  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf674, (6, 128, 64), (64, 384, 1), 0), permute_897, out=buf676)
    del permute_897
    buf677 = buf639; del buf639  # reuse
    buf678 = buf676; del buf676  # reuse
    buf679 = reinterpret_tensor(buf640, (98304, ), (1, ), 0); del buf640  # reuse
    buf682 = reinterpret_tensor(buf449, (6, 128, 128), (16384, 128, 1), 0); del buf449  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_133(c_void_p(buf678.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(alias_148.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf682.data_ptr()))
    del alias_148
    del getitem_35
    buf684 = reinterpret_tensor(buf674, (6, 64, 128), (8192, 128, 1), 0); del buf674  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_898, buf682, out=buf684)
    del permute_898
    buf685 = reinterpret_tensor(buf648, (6, 128, 64), (8192, 64, 1), 0); del buf648  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf682, permute_899, out=buf685)
    del permute_899
    buf686 = reinterpret_tensor(buf637, (128, 384), (384, 1), 0); del buf637  # reuse
    cpp_fused_view_134(c_void_p(buf675.data_ptr()), c_void_p(buf686.data_ptr()))
    buf687 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf686, (384, 128), (1, 384), 0), view_105, out=buf687)
    buf688 = reinterpret_tensor(buf672, (128, 512), (512, 1), 0); del buf672  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf686, permute_904, out=buf688)
    del permute_904
    buf689 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf684, (384, 128), (128, 1), 0), view_105, out=buf689)
    buf690 = buf668; del buf668  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf684, (128, 384), (1, 128), 0), permute_909, out=buf690)
    del permute_909
    buf691 = reinterpret_tensor(buf684, (128, 384), (384, 1), 0); del buf684  # reuse
    cpp_fused_view_135(c_void_p(buf685.data_ptr()), c_void_p(buf691.data_ptr()))
    buf692 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf691, (384, 128), (1, 384), 0), view_105, out=buf692)
    del view_105
    buf693 = buf664; del buf664  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf691, permute_914, out=buf693)
    del permute_914
    buf694 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf695 = buf670; del buf670  # reuse
    buf696 = buf671; del buf671  # reuse
    buf697 = reinterpret_tensor(buf650, (1, 128, 512), (65536, 512, 1), 0); del buf650  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_136(c_void_p(buf696.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(add_31.data_ptr()), c_void_p(rsqrt_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf697.data_ptr()))
    del add_31
    del getitem_33
    del primals_9
    del rsqrt_8
    buf698 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf697, (512, 128), (1, 512), 0), view_103, out=buf698)
    del view_103
    buf699 = reinterpret_tensor(buf666, (128, 1024), (1024, 1), 0); del buf666  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf697, (128, 512), (512, 1), 0), permute_918, out=buf699)
    del permute_918
    buf700 = buf662; del buf662  # reuse
    buf703 = reinterpret_tensor(buf661, (1, 128, 1024), (131072, 1024, 1), 0); del buf661  # reuse
    buf704 = buf703; del buf703  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_137(c_void_p(buf704.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(mm_25.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(mm_26.data_ptr()), c_void_p(buf700.data_ptr()))
    del getitem_31
    del mm_25
    del mm_26
    del tanh_3
    buf701 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf700, (1024, 128), (1, 1024), 0), view_99, out=buf701)
    buf702 = reinterpret_tensor(buf697, (128, 512), (512, 1), 0); del buf697  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf700, (128, 1024), (1024, 1), 0), permute_922, out=buf702)
    del permute_922
    buf705 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf704, (1024, 128), (1, 1024), 0), view_99, out=buf705)
    del view_99
    buf706 = buf693; del buf693  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf704, (128, 1024), (1024, 1), 0), permute_926, out=buf706)
    del permute_926
    buf707 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf708 = buf695; del buf695  # reuse
    buf709 = buf696; del buf696  # reuse
    buf710 = reinterpret_tensor(buf690, (1, 128, 512), (65536, 512, 1), 0); del buf690  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_138(c_void_p(buf709.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(add_27.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf710.data_ptr()))
    del add_27
    del getitem_29
    del primals_8
    del rsqrt_7
    buf711 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf710, (512, 128), (1, 512), 0), view_97, out=buf711)
    del view_97
    buf712 = buf691; del buf691  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf710, (128, 512), (512, 1), 0), permute_930, out=buf712)
    del permute_930
    buf713 = buf685; del buf685  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_933, reinterpret_tensor(buf712, (6, 128, 64), (64, 384, 1), 0), out=buf713)
    del permute_933
    buf714 = buf682; del buf682  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf712, (6, 128, 64), (64, 384, 1), 0), permute_934, out=buf714)
    del permute_934
    buf715 = buf677; del buf677  # reuse
    buf716 = buf714; del buf714  # reuse
    buf717 = reinterpret_tensor(buf678, (98304, ), (1, ), 0); del buf678  # reuse
    buf720 = reinterpret_tensor(buf384, (6, 128, 128), (16384, 128, 1), 0); del buf384  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_139(c_void_p(buf716.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(alias_152.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf720.data_ptr()))
    del alias_152
    del getitem_27
    buf722 = reinterpret_tensor(buf712, (6, 64, 128), (8192, 128, 1), 0); del buf712  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_935, buf720, out=buf722)
    del permute_935
    buf723 = reinterpret_tensor(buf686, (6, 128, 64), (8192, 64, 1), 0); del buf686  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf720, permute_936, out=buf723)
    del permute_936
    buf724 = reinterpret_tensor(buf675, (128, 384), (384, 1), 0); del buf675  # reuse
    cpp_fused_view_140(c_void_p(buf713.data_ptr()), c_void_p(buf724.data_ptr()))
    buf725 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf724, (384, 128), (1, 384), 0), view_79, out=buf725)
    buf726 = reinterpret_tensor(buf710, (128, 512), (512, 1), 0); del buf710  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf724, permute_941, out=buf726)
    del permute_941
    buf727 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf722, (384, 128), (128, 1), 0), view_79, out=buf727)
    buf728 = buf706; del buf706  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf722, (128, 384), (1, 128), 0), permute_946, out=buf728)
    del permute_946
    buf729 = reinterpret_tensor(buf722, (128, 384), (384, 1), 0); del buf722  # reuse
    cpp_fused_view_141(c_void_p(buf723.data_ptr()), c_void_p(buf729.data_ptr()))
    buf730 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf729, (384, 128), (1, 384), 0), view_79, out=buf730)
    del view_79
    buf731 = buf702; del buf702  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf729, permute_951, out=buf731)
    del permute_951
    buf732 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf733 = buf708; del buf708  # reuse
    buf734 = buf709; del buf709  # reuse
    buf735 = reinterpret_tensor(buf688, (1, 128, 512), (65536, 512, 1), 0); del buf688  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_142(c_void_p(buf734.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf735.data_ptr()))
    del add_24
    del getitem_25
    del primals_7
    del rsqrt_6
    buf736 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf735, (512, 128), (1, 512), 0), view_77, out=buf736)
    del view_77
    buf737 = reinterpret_tensor(buf704, (128, 1024), (1024, 1), 0); del buf704  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf735, (128, 512), (512, 1), 0), permute_955, out=buf737)
    del permute_955
    buf738 = buf700; del buf700  # reuse
    buf741 = reinterpret_tensor(buf699, (1, 128, 1024), (131072, 1024, 1), 0); del buf699  # reuse
    buf742 = buf741; del buf741  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_143(c_void_p(buf742.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(mm_18.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(mm_19.data_ptr()), c_void_p(buf738.data_ptr()))
    del getitem_23
    del mm_18
    del mm_19
    del tanh_2
    buf739 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf738, (1024, 128), (1, 1024), 0), view_73, out=buf739)
    buf740 = reinterpret_tensor(buf735, (128, 512), (512, 1), 0); del buf735  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf738, (128, 1024), (1024, 1), 0), permute_959, out=buf740)
    del permute_959
    buf743 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf742, (1024, 128), (1, 1024), 0), view_73, out=buf743)
    del view_73
    buf744 = buf731; del buf731  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf742, (128, 1024), (1024, 1), 0), permute_963, out=buf744)
    del permute_963
    buf745 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf746 = buf733; del buf733  # reuse
    buf747 = buf734; del buf734  # reuse
    buf748 = reinterpret_tensor(buf728, (1, 128, 512), (65536, 512, 1), 0); del buf728  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_144(c_void_p(buf747.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(add_20.data_ptr()), c_void_p(rsqrt_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf748.data_ptr()))
    del add_20
    del getitem_21
    del primals_6
    del rsqrt_5
    buf749 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (512, 128), (1, 512), 0), view_71, out=buf749)
    del view_71
    buf750 = buf729; del buf729  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf748, (128, 512), (512, 1), 0), permute_967, out=buf750)
    del permute_967
    buf751 = buf723; del buf723  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_970, reinterpret_tensor(buf750, (6, 128, 64), (64, 384, 1), 0), out=buf751)
    del permute_970
    buf752 = buf720; del buf720  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf750, (6, 128, 64), (64, 384, 1), 0), permute_971, out=buf752)
    del permute_971
    buf753 = buf715; del buf715  # reuse
    buf754 = buf752; del buf752  # reuse
    buf755 = reinterpret_tensor(buf716, (98304, ), (1, ), 0); del buf716  # reuse
    buf758 = reinterpret_tensor(buf319, (6, 128, 128), (16384, 128, 1), 0); del buf319  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_145(c_void_p(buf754.data_ptr()), c_void_p(getitem_19.data_ptr()), c_void_p(alias_156.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf758.data_ptr()))
    del alias_156
    del getitem_19
    buf760 = reinterpret_tensor(buf750, (6, 64, 128), (8192, 128, 1), 0); del buf750  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_972, buf758, out=buf760)
    del permute_972
    buf761 = reinterpret_tensor(buf724, (6, 128, 64), (8192, 64, 1), 0); del buf724  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf758, permute_973, out=buf761)
    del permute_973
    buf762 = reinterpret_tensor(buf713, (128, 384), (384, 1), 0); del buf713  # reuse
    cpp_fused_view_146(c_void_p(buf751.data_ptr()), c_void_p(buf762.data_ptr()))
    buf763 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf762, (384, 128), (1, 384), 0), view_53, out=buf763)
    buf764 = reinterpret_tensor(buf748, (128, 512), (512, 1), 0); del buf748  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf762, permute_978, out=buf764)
    del permute_978
    buf765 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf760, (384, 128), (128, 1), 0), view_53, out=buf765)
    buf766 = buf744; del buf744  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf760, (128, 384), (1, 128), 0), permute_983, out=buf766)
    del permute_983
    buf767 = reinterpret_tensor(buf760, (128, 384), (384, 1), 0); del buf760  # reuse
    cpp_fused_view_147(c_void_p(buf761.data_ptr()), c_void_p(buf767.data_ptr()))
    buf768 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf767, (384, 128), (1, 384), 0), view_53, out=buf768)
    del view_53
    buf769 = buf740; del buf740  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf767, permute_988, out=buf769)
    del permute_988
    buf770 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf771 = buf746; del buf746  # reuse
    buf772 = buf747; del buf747  # reuse
    buf773 = reinterpret_tensor(buf726, (1, 128, 512), (65536, 512, 1), 0); del buf726  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_148(c_void_p(buf772.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(add_17.data_ptr()), c_void_p(rsqrt_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf773.data_ptr()))
    del add_17
    del getitem_17
    del primals_5
    del rsqrt_4
    buf774 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf773, (512, 128), (1, 512), 0), view_51, out=buf774)
    del view_51
    buf775 = reinterpret_tensor(buf742, (128, 1024), (1024, 1), 0); del buf742  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf773, (128, 512), (512, 1), 0), permute_992, out=buf775)
    del permute_992
    buf776 = buf738; del buf738  # reuse
    buf779 = reinterpret_tensor(buf737, (1, 128, 1024), (131072, 1024, 1), 0); del buf737  # reuse
    buf780 = buf779; del buf779  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_149(c_void_p(buf780.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(getitem_15.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(mm_12.data_ptr()), c_void_p(buf776.data_ptr()))
    del getitem_15
    del mm_11
    del mm_12
    del tanh_1
    buf777 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf776, (1024, 128), (1, 1024), 0), view_47, out=buf777)
    buf778 = reinterpret_tensor(buf773, (128, 512), (512, 1), 0); del buf773  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf776, (128, 1024), (1024, 1), 0), permute_996, out=buf778)
    del permute_996
    buf781 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf780, (1024, 128), (1, 1024), 0), view_47, out=buf781)
    del view_47
    buf782 = buf769; del buf769  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf780, (128, 1024), (1024, 1), 0), permute_1000, out=buf782)
    del permute_1000
    buf783 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf784 = buf771; del buf771  # reuse
    buf785 = buf772; del buf772  # reuse
    buf786 = reinterpret_tensor(buf766, (1, 128, 512), (65536, 512, 1), 0); del buf766  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_150(c_void_p(buf785.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(add_13.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf786.data_ptr()))
    del add_13
    del getitem_13
    del primals_4
    del rsqrt_3
    buf787 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf786, (512, 128), (1, 512), 0), view_45, out=buf787)
    del view_45
    buf788 = buf767; del buf767  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf786, (128, 512), (512, 1), 0), permute_1004, out=buf788)
    del permute_1004
    buf789 = buf761; del buf761  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1007, reinterpret_tensor(buf788, (6, 128, 64), (64, 384, 1), 0), out=buf789)
    del permute_1007
    buf790 = buf758; del buf758  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf788, (6, 128, 64), (64, 384, 1), 0), permute_1008, out=buf790)
    del permute_1008
    buf791 = buf753; del buf753  # reuse
    buf792 = buf790; del buf790  # reuse
    buf793 = reinterpret_tensor(buf754, (98304, ), (1, ), 0); del buf754  # reuse
    buf796 = reinterpret_tensor(buf254, (6, 128, 128), (16384, 128, 1), 0); del buf254  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_native_dropout_backward_squeeze_151(c_void_p(buf792.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(alias_160.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf796.data_ptr()))
    del alias_160
    del getitem_11
    buf798 = reinterpret_tensor(buf788, (6, 64, 128), (8192, 128, 1), 0); del buf788  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1009, buf796, out=buf798)
    del permute_1009
    buf799 = reinterpret_tensor(buf762, (6, 128, 64), (8192, 64, 1), 0); del buf762  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf796, permute_1010, out=buf799)
    del permute_1010
    buf800 = reinterpret_tensor(buf751, (128, 384), (384, 1), 0); del buf751  # reuse
    cpp_fused_view_152(c_void_p(buf789.data_ptr()), c_void_p(buf800.data_ptr()))
    buf801 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf800, (384, 128), (1, 384), 0), view_27, out=buf801)
    buf802 = reinterpret_tensor(buf786, (128, 512), (512, 1), 0); del buf786  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf800, permute_1015, out=buf802)
    del permute_1015
    buf803 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf798, (384, 128), (128, 1), 0), view_27, out=buf803)
    buf804 = buf782; del buf782  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf798, (128, 384), (1, 128), 0), permute_1020, out=buf804)
    del permute_1020
    buf805 = reinterpret_tensor(buf798, (128, 384), (384, 1), 0); del buf798  # reuse
    cpp_fused_view_153(c_void_p(buf799.data_ptr()), c_void_p(buf805.data_ptr()))
    buf806 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf805, (384, 128), (1, 384), 0), view_27, out=buf806)
    del view_27
    buf807 = buf778; del buf778  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf805, permute_1025, out=buf807)
    del permute_1025
    buf808 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf809 = buf784; del buf784  # reuse
    buf810 = buf785; del buf785  # reuse
    buf811 = reinterpret_tensor(buf764, (1, 128, 512), (65536, 512, 1), 0); del buf764  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_154(c_void_p(buf810.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(add_10.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(getitem_9.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf811.data_ptr()))
    del add_10
    del buf802
    del getitem_9
    del primals_3
    del rsqrt_2
    buf812 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf811, (512, 128), (1, 512), 0), view_25, out=buf812)
    del view_25
    buf813 = reinterpret_tensor(buf780, (128, 1024), (1024, 1), 0); del buf780  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf811, (128, 512), (512, 1), 0), permute_1029, out=buf813)
    del permute_1029
    buf814 = buf776; del buf776  # reuse
    buf817 = reinterpret_tensor(buf775, (1, 128, 1024), (131072, 1024, 1), 0); del buf775  # reuse
    buf818 = buf817; del buf817  # reuse
    cpp_fused_add_mul_native_dropout_backward_pow_tanh_backward_155(c_void_p(buf818.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(mm_4.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(buf814.data_ptr()))
    del buf813
    del getitem_7
    del mm_4
    del mm_5
    del tanh
    buf815 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf814, (1024, 128), (1, 1024), 0), view_21, out=buf815)
    buf816 = reinterpret_tensor(buf811, (128, 512), (512, 1), 0); del buf811  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf814, (128, 1024), (1024, 1), 0), permute_1033, out=buf816)
    del buf814
    del permute_1033
    buf819 = empty((1024, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf818, (1024, 128), (1, 1024), 0), view_21, out=buf819)
    del view_21
    buf820 = buf807; del buf807  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf818, (128, 1024), (1024, 1), 0), permute_1037, out=buf820)
    del buf818
    del permute_1037
    buf821 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf822 = buf809; del buf809  # reuse
    buf823 = buf810; del buf810  # reuse
    buf824 = reinterpret_tensor(buf804, (1, 128, 512), (65536, 512, 1), 0); del buf804  # reuse
    cpp_fused_add_div_mul_native_dropout_backward_pow_sum_156(c_void_p(buf823.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(add_6.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf824.data_ptr()))
    del add_6
    del getitem_5
    del primals_2
    del rsqrt_1
    buf825 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf824, (512, 128), (1, 512), 0), view_19, out=buf825)
    del view_19
    buf826 = buf805; del buf805  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf824, (128, 512), (512, 1), 0), permute_1041, out=buf826)
    del permute_1041
    buf827 = buf799; del buf799  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1044, reinterpret_tensor(buf826, (6, 128, 64), (64, 384, 1), 0), out=buf827)
    del permute_1044
    buf828 = buf796; del buf796  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf826, (6, 128, 64), (64, 384, 1), 0), permute_1045, out=buf828)
    del permute_1045
    buf829 = buf791; del buf791  # reuse
    buf830 = buf828; del buf828  # reuse
    buf831 = reinterpret_tensor(buf792, (98304, ), (1, ), 0); del buf792  # reuse
    buf834 = reinterpret_tensor(buf188, (6, 128, 128), (16384, 128, 1), 0); del buf188  # reuse
    buf837 = reinterpret_tensor(buf123, (128, 128, 6), (128, 1, 16384), 0); del buf123  # reuse
    buf836 = empty((32, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_embedding_dense_backward_native_dropout_backward_nll_loss_forward_squeeze_157(c_void_p(buf830.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(alias_164.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf836.data_ptr()))
    del alias_164
    del buf565
    del buf603
    del buf641
    del buf679
    del buf717
    del buf755
    del buf793
    del buf829
    del buf830
    del buf831
    del getitem_3
    aten.index_put_(buf836, [add_3], buf837, True)
    del add_3
    del buf837
    buf840 = reinterpret_tensor(buf826, (6, 64, 128), (8192, 128, 1), 0); del buf826  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1047, buf834, out=buf840)
    del permute_1047
    buf841 = reinterpret_tensor(buf800, (6, 128, 64), (8192, 64, 1), 0); del buf800  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf834, permute_1048, out=buf841)
    del buf834
    del permute_1048
    buf842 = reinterpret_tensor(buf789, (128, 384), (384, 1), 0); del buf789  # reuse
    cpp_fused_view_158(c_void_p(buf827.data_ptr()), c_void_p(buf842.data_ptr()))
    del buf827
    buf843 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf842, (384, 128), (1, 384), 0), view_1, out=buf843)
    buf844 = reinterpret_tensor(buf824, (128, 512), (512, 1), 0); del buf824  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf842, permute_1053, out=buf844)
    del buf842
    del permute_1053
    buf845 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf840, (384, 128), (128, 1), 0), view_1, out=buf845)
    buf846 = buf820; del buf820  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf840, (128, 384), (1, 128), 0), permute_1058, out=buf846)
    del permute_1058
    buf847 = reinterpret_tensor(buf840, (128, 384), (384, 1), 0); del buf840  # reuse
    cpp_fused_view_159(c_void_p(buf841.data_ptr()), c_void_p(buf847.data_ptr()))
    del buf841
    buf848 = empty((384, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf847, (384, 128), (1, 384), 0), view_1, out=buf848)
    del view_1
    buf849 = buf816; del buf816  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf847, permute_1063, out=buf849)
    del buf847
    del permute_1063
    buf850 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf851 = buf822; del buf822  # reuse
    buf852 = buf823; del buf823  # reuse
    buf854 = buf852; del buf852  # reuse
    buf853 = empty((250112, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_native_dropout_backward_nll_loss_forward_pow_sum_160(c_void_p(buf854.data_ptr()), c_void_p(buf844.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(getitem.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf853.data_ptr()))
    del buf844
    del buf846
    del buf849
    del buf851
    del getitem
    del getitem_1
    del primals_1
    del rsqrt
    aten.index_put_(buf853, [view], buf854, True)
    del buf854
    del view
    buf541 = empty((250112, 512), device='cpu', dtype=torch.float32)
    buf857 = buf541; del buf541  # reuse
    cpp_fused_add_161(c_void_p(buf857.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf853.data_ptr()))
    return (reinterpret_tensor(buf850, (512, ), (1, ), 0), reinterpret_tensor(buf821, (512, ), (1, ), 0), reinterpret_tensor(buf808, (512, ), (1, ), 0), reinterpret_tensor(buf783, (512, ), (1, ), 0), reinterpret_tensor(buf770, (512, ), (1, ), 0), reinterpret_tensor(buf745, (512, ), (1, ), 0), reinterpret_tensor(buf732, (512, ), (1, ), 0), reinterpret_tensor(buf707, (512, ), (1, ), 0), reinterpret_tensor(buf694, (512, ), (1, ), 0), reinterpret_tensor(buf669, (512, ), (1, ), 0), reinterpret_tensor(buf656, (512, ), (1, ), 0), reinterpret_tensor(buf631, (512, ), (1, ), 0), reinterpret_tensor(buf618, (512, ), (1, ), 0), reinterpret_tensor(buf593, (512, ), (1, ), 0), reinterpret_tensor(buf580, (512, ), (1, ), 0), reinterpret_tensor(buf555, (512, ), (1, ), 0), reinterpret_tensor(buf542, (512, ), (1, ), 0), reinterpret_tensor(buf535, (512, ), (1, ), 0), reinterpret_tensor(buf505, (512, ), (1, ), 0), reinterpret_tensor(buf478, (512, ), (1, ), 0), reinterpret_tensor(buf465, (512, ), (1, ), 0), reinterpret_tensor(buf439, (512, ), (1, ), 0), reinterpret_tensor(buf413, (512, ), (1, ), 0), reinterpret_tensor(buf400, (512, ), (1, ), 0), reinterpret_tensor(buf374, (512, ), (1, ), 0), reinterpret_tensor(buf348, (512, ), (1, ), 0), reinterpret_tensor(buf335, (512, ), (1, ), 0), reinterpret_tensor(buf309, (512, ), (1, ), 0), reinterpret_tensor(buf283, (512, ), (1, ), 0), reinterpret_tensor(buf270, (512, ), (1, ), 0), reinterpret_tensor(buf244, (512, ), (1, ), 0), reinterpret_tensor(buf217, (512, ), (1, ), 0), reinterpret_tensor(buf204, (512, ), (1, ), 0), reinterpret_tensor(buf178, (512, ), (1, ), 0), reinterpret_tensor(buf152, (512, ), (1, ), 0), reinterpret_tensor(buf139, (512, ), (1, ), 0), reinterpret_tensor(buf113, (512, ), (1, ), 0), reinterpret_tensor(buf87, (512, ), (1, ), 0), reinterpret_tensor(buf74, (512, ), (1, ), 0), reinterpret_tensor(buf48, (512, ), (1, ), 0), reinterpret_tensor(buf21, (512, ), (1, ), 0), reinterpret_tensor(buf8, (512, ), (1, ), 0), buf857, reinterpret_tensor(buf848, (384, 512), (512, 1), 0), reinterpret_tensor(buf845, (384, 512), (512, 1), 0), reinterpret_tensor(buf843, (384, 512), (512, 1), 0), buf836, reinterpret_tensor(buf825, (512, 384), (384, 1), 0), reinterpret_tensor(buf819, (1024, 512), (512, 1), 0), reinterpret_tensor(buf815, (1024, 512), (512, 1), 0), reinterpret_tensor(buf812, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf806, (384, 512), (512, 1), 0), reinterpret_tensor(buf803, (384, 512), (512, 1), 0), reinterpret_tensor(buf801, (384, 512), (512, 1), 0), reinterpret_tensor(buf787, (512, 384), (384, 1), 0), reinterpret_tensor(buf781, (1024, 512), (512, 1), 0), reinterpret_tensor(buf777, (1024, 512), (512, 1), 0), reinterpret_tensor(buf774, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf768, (384, 512), (512, 1), 0), reinterpret_tensor(buf765, (384, 512), (512, 1), 0), reinterpret_tensor(buf763, (384, 512), (512, 1), 0), reinterpret_tensor(buf749, (512, 384), (384, 1), 0), reinterpret_tensor(buf743, (1024, 512), (512, 1), 0), reinterpret_tensor(buf739, (1024, 512), (512, 1), 0), reinterpret_tensor(buf736, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf730, (384, 512), (512, 1), 0), reinterpret_tensor(buf727, (384, 512), (512, 1), 0), reinterpret_tensor(buf725, (384, 512), (512, 1), 0), reinterpret_tensor(buf711, (512, 384), (384, 1), 0), reinterpret_tensor(buf705, (1024, 512), (512, 1), 0), reinterpret_tensor(buf701, (1024, 512), (512, 1), 0), reinterpret_tensor(buf698, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf692, (384, 512), (512, 1), 0), reinterpret_tensor(buf689, (384, 512), (512, 1), 0), reinterpret_tensor(buf687, (384, 512), (512, 1), 0), reinterpret_tensor(buf673, (512, 384), (384, 1), 0), reinterpret_tensor(buf667, (1024, 512), (512, 1), 0), reinterpret_tensor(buf663, (1024, 512), (512, 1), 0), reinterpret_tensor(buf660, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf654, (384, 512), (512, 1), 0), reinterpret_tensor(buf651, (384, 512), (512, 1), 0), reinterpret_tensor(buf649, (384, 512), (512, 1), 0), reinterpret_tensor(buf635, (512, 384), (384, 1), 0), reinterpret_tensor(buf629, (1024, 512), (512, 1), 0), reinterpret_tensor(buf625, (1024, 512), (512, 1), 0), reinterpret_tensor(buf622, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf616, (384, 512), (512, 1), 0), reinterpret_tensor(buf613, (384, 512), (512, 1), 0), reinterpret_tensor(buf611, (384, 512), (512, 1), 0), reinterpret_tensor(buf597, (512, 384), (384, 1), 0), reinterpret_tensor(buf591, (1024, 512), (512, 1), 0), reinterpret_tensor(buf587, (1024, 512), (512, 1), 0), reinterpret_tensor(buf584, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf578, (384, 512), (512, 1), 0), reinterpret_tensor(buf575, (384, 512), (512, 1), 0), reinterpret_tensor(buf573, (384, 512), (512, 1), 0), reinterpret_tensor(buf559, (512, 384), (384, 1), 0), reinterpret_tensor(buf553, (1024, 512), (512, 1), 0), reinterpret_tensor(buf549, (1024, 512), (512, 1), 0), reinterpret_tensor(buf546, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf533, (384, 512), (512, 1), 0), reinterpret_tensor(buf530, (384, 512), (512, 1), 0), reinterpret_tensor(buf527, (384, 512), (512, 1), 0), buf520, reinterpret_tensor(buf509, (512, 384), (384, 1), 0), reinterpret_tensor(buf503, (384, 512), (512, 1), 0), reinterpret_tensor(buf499, (384, 512), (512, 1), 0), reinterpret_tensor(buf496, (384, 512), (512, 1), 0), reinterpret_tensor(buf482, (512, 384), (384, 1), 0), reinterpret_tensor(buf476, (1024, 512), (512, 1), 0), reinterpret_tensor(buf472, (1024, 512), (512, 1), 0), reinterpret_tensor(buf469, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf463, (384, 512), (512, 1), 0), reinterpret_tensor(buf460, (384, 512), (512, 1), 0), reinterpret_tensor(buf457, (384, 512), (512, 1), 0), reinterpret_tensor(buf443, (512, 384), (384, 1), 0), reinterpret_tensor(buf437, (384, 512), (512, 1), 0), reinterpret_tensor(buf434, (384, 512), (512, 1), 0), reinterpret_tensor(buf431, (384, 512), (512, 1), 0), reinterpret_tensor(buf417, (512, 384), (384, 1), 0), reinterpret_tensor(buf411, (1024, 512), (512, 1), 0), reinterpret_tensor(buf407, (1024, 512), (512, 1), 0), reinterpret_tensor(buf404, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf398, (384, 512), (512, 1), 0), reinterpret_tensor(buf395, (384, 512), (512, 1), 0), reinterpret_tensor(buf392, (384, 512), (512, 1), 0), reinterpret_tensor(buf378, (512, 384), (384, 1), 0), reinterpret_tensor(buf372, (384, 512), (512, 1), 0), reinterpret_tensor(buf369, (384, 512), (512, 1), 0), reinterpret_tensor(buf366, (384, 512), (512, 1), 0), reinterpret_tensor(buf352, (512, 384), (384, 1), 0), reinterpret_tensor(buf346, (1024, 512), (512, 1), 0), reinterpret_tensor(buf342, (1024, 512), (512, 1), 0), reinterpret_tensor(buf339, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf333, (384, 512), (512, 1), 0), reinterpret_tensor(buf330, (384, 512), (512, 1), 0), reinterpret_tensor(buf327, (384, 512), (512, 1), 0), reinterpret_tensor(buf313, (512, 384), (384, 1), 0), reinterpret_tensor(buf307, (384, 512), (512, 1), 0), reinterpret_tensor(buf304, (384, 512), (512, 1), 0), reinterpret_tensor(buf301, (384, 512), (512, 1), 0), reinterpret_tensor(buf287, (512, 384), (384, 1), 0), reinterpret_tensor(buf281, (1024, 512), (512, 1), 0), reinterpret_tensor(buf277, (1024, 512), (512, 1), 0), reinterpret_tensor(buf274, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf268, (384, 512), (512, 1), 0), reinterpret_tensor(buf265, (384, 512), (512, 1), 0), reinterpret_tensor(buf262, (384, 512), (512, 1), 0), reinterpret_tensor(buf248, (512, 384), (384, 1), 0), reinterpret_tensor(buf242, (384, 512), (512, 1), 0), reinterpret_tensor(buf238, (384, 512), (512, 1), 0), reinterpret_tensor(buf235, (384, 512), (512, 1), 0), reinterpret_tensor(buf221, (512, 384), (384, 1), 0), reinterpret_tensor(buf215, (1024, 512), (512, 1), 0), reinterpret_tensor(buf211, (1024, 512), (512, 1), 0), reinterpret_tensor(buf208, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf202, (384, 512), (512, 1), 0), reinterpret_tensor(buf199, (384, 512), (512, 1), 0), reinterpret_tensor(buf196, (384, 512), (512, 1), 0), reinterpret_tensor(buf182, (512, 384), (384, 1), 0), reinterpret_tensor(buf176, (384, 512), (512, 1), 0), reinterpret_tensor(buf173, (384, 512), (512, 1), 0), reinterpret_tensor(buf170, (384, 512), (512, 1), 0), reinterpret_tensor(buf156, (512, 384), (384, 1), 0), reinterpret_tensor(buf150, (1024, 512), (512, 1), 0), reinterpret_tensor(buf146, (1024, 512), (512, 1), 0), reinterpret_tensor(buf143, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf137, (384, 512), (512, 1), 0), reinterpret_tensor(buf134, (384, 512), (512, 1), 0), reinterpret_tensor(buf131, (384, 512), (512, 1), 0), reinterpret_tensor(buf117, (512, 384), (384, 1), 0), reinterpret_tensor(buf111, (384, 512), (512, 1), 0), reinterpret_tensor(buf108, (384, 512), (512, 1), 0), reinterpret_tensor(buf105, (384, 512), (512, 1), 0), reinterpret_tensor(buf91, (512, 384), (384, 1), 0), reinterpret_tensor(buf85, (1024, 512), (512, 1), 0), reinterpret_tensor(buf81, (1024, 512), (512, 1), 0), reinterpret_tensor(buf78, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (384, 512), (512, 1), 0), reinterpret_tensor(buf69, (384, 512), (512, 1), 0), reinterpret_tensor(buf66, (384, 512), (512, 1), 0), reinterpret_tensor(buf52, (512, 384), (384, 1), 0), reinterpret_tensor(buf46, (384, 512), (512, 1), 0), reinterpret_tensor(buf43, (384, 512), (512, 1), 0), reinterpret_tensor(buf40, (384, 512), (512, 1), 0), reinterpret_tensor(buf25, (512, 384), (384, 1), 0), reinterpret_tensor(buf19, (1024, 512), (512, 1), 0), reinterpret_tensor(buf15, (1024, 512), (512, 1), 0), reinterpret_tensor(buf12, (512, 1024), (1024, 1), 0), reinterpret_tensor(buf6, (250112, 512), (512, 1), 0), None, None, None, )


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
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    view = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    getitem = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_3 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.int64)
    getitem_3 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_19 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_6 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_4 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_25 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_10 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_45 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_13 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_12 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_15 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_51 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_17 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_4 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_71 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_20 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_5 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_18 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_19 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_77 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_24 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_97 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_27 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_7 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_25 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_26 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_103 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_31 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_8 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_123 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_34 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_9 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_32 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_33 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_129 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_38 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_10 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_131 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_149 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_41 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_11 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_151 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_39 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_40 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_155 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_45 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_12 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_157 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_175 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_48 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_13 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_46 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_47 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_181 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_52 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_14 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_183 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_59 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_201 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_55 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_15 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_203 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_53 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_54 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_207 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_65 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_59 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    view_209 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    getitem_68 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    getitem_69 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    rsqrt_17 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_210 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_63 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.int64)
    getitem_71 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_228 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_66 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_230 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_233 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_75 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_248 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_70 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_250 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_64 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_65 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_254 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_74 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_20 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_256 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_274 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_85 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_77 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_276 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_294 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_89 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_80 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_296 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_75 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_76 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_300 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_93 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_84 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_23 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_302 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_95 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_320 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_87 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_322 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_340 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_90 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_342 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_86 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_87 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_346 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_105 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_94 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_26 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_348 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_366 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_109 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_97 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_368 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_386 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_113 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_100 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_388 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_97 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_98 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_392 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_104 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_29 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_394 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_412 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_107 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_414 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_123 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_432 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_125 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_110 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_434 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_108 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_12 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_109 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_438 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_129 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_114 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_32 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_440 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_458 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_133 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_117 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_460 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_478 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_137 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_120 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_480 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_119 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_13 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_120 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_484 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_124 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_35 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_486 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_143 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_504 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_127 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_506 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_524 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_149 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_130 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_37 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_526 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_130 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_14 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_131 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_530 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_153 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_134 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_532 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_155 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_550 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_157 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_137 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_552 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    getitem_159 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_570 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_161 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_140 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    view_572 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_141 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    tanh_15 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    mm_142 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_163 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.bool)
    view_576 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_165 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    add_144 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    getitem_167 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.bool)
    view_578 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    sub_30 = rand_strided((128, 250112), (250112, 1), device='cpu', dtype=torch.float32)
    convert_element_type_7 = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((250112, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_281 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_87 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_296 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_301 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_314 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_89 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_321 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_326 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_331 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_339 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_347 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_351 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_93 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_352 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_353 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_363 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_375 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_95 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_378 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_397 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_401 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_412 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_99 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_425 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_437 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_101 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_445 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_463 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_474 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_105 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_477 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_487 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_496 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_499 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_107 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_501 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_502 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_507 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_512 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_517 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_521 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_529 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_533 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_536 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_537 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_111 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_538 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_539 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_544 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_549 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_554 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_558 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_561 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_562 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_113 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_563 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_564 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_569 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_574 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_579 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_583 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_587 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_591 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_595 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_598 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_599 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_117 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_600 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_601 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_606 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_611 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_616 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_620 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_623 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_624 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_119 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_625 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_626 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_631 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_636 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_641 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_645 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_649 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_653 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_657 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_660 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_661 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_123 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_662 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_663 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_668 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_673 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_678 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_682 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_685 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_686 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_125 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_687 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_688 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_693 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_698 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_703 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_707 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_711 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_715 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_719 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_722 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_723 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_129 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_724 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_725 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_730 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_735 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_740 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_744 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_747 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_748 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_131 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_750 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_751 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_756 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_761 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_766 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_770 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_774 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_778 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_782 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_785 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_786 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_136 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_787 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_788 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_793 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_798 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_803 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_807 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_811 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_815 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_819 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_822 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_823 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_140 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_824 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_825 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_830 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_835 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_840 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_844 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_848 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_852 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_856 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_859 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_860 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_144 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_861 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_862 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_867 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_872 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_877 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_881 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_885 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_889 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_893 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_896 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_897 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_148 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_898 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_899 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_904 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_909 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_914 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_918 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_922 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_926 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_930 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_933 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_934 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_152 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_935 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_936 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_941 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_946 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_951 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_955 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_959 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_963 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_967 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_970 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_971 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_156 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_972 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_973 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_978 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_983 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_988 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_992 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_996 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1000 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1004 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_1007 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1008 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_160 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1009 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_1010 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_1015 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1020 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1025 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1029 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1033 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1037 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1041 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_1044 = rand_strided((6, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_1045 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    alias_164 = rand_strided((1, 6, 128, 128), (98304, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_1047 = rand_strided((6, 64, 128), (64, 1, 384), device='cpu', dtype=torch.float32)
    permute_1048 = rand_strided((6, 128, 64), (64, 384, 1), device='cpu', dtype=torch.float32)
    permute_1053 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1058 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_1063 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128, 250112), (32014336, 250112, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_27 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_28 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_29 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_30 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_31 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_32 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_33 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_34 = rand_strided((1, 6, 128, 64), (49152, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_35 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, mm_4, tanh, mm_5, getitem_7, view_25, getitem_9, add_10, rsqrt_2, view_27, getitem_11, view_45, getitem_13, add_13, rsqrt_3, view_47, mm_11, tanh_1, mm_12, getitem_15, view_51, getitem_17, add_17, rsqrt_4, view_53, getitem_19, view_71, getitem_21, add_20, rsqrt_5, view_73, mm_18, tanh_2, mm_19, getitem_23, view_77, getitem_25, add_24, rsqrt_6, view_79, getitem_27, view_97, getitem_29, add_27, rsqrt_7, view_99, mm_25, tanh_3, mm_26, getitem_31, view_103, getitem_33, add_31, rsqrt_8, view_105, getitem_35, view_123, getitem_37, add_34, rsqrt_9, view_125, mm_32, tanh_4, mm_33, getitem_39, view_129, getitem_41, add_38, rsqrt_10, view_131, getitem_43, view_149, getitem_45, add_41, rsqrt_11, view_151, mm_39, tanh_5, mm_40, getitem_47, view_155, getitem_49, add_45, rsqrt_12, view_157, getitem_51, view_175, getitem_53, add_48, rsqrt_13, view_177, mm_46, tanh_6, mm_47, getitem_55, view_181, getitem_57, add_52, rsqrt_14, view_183, getitem_59, view_201, getitem_61, add_55, rsqrt_15, view_203, mm_53, tanh_7, mm_54, getitem_63, view_207, getitem_65, add_59, rsqrt_16, getitem_67, view_209, getitem_68, getitem_69, rsqrt_17, view_210, add_63, getitem_71, view_228, getitem_73, add_66, rsqrt_18, view_230, view_233, getitem_75, view_248, getitem_77, add_70, rsqrt_19, view_250, mm_64, tanh_8, mm_65, getitem_79, view_254, getitem_81, add_74, rsqrt_20, view_256, getitem_83, view_274, getitem_85, add_77, rsqrt_21, view_276, getitem_87, view_294, getitem_89, add_80, rsqrt_22, view_296, mm_75, tanh_9, mm_76, getitem_91, view_300, getitem_93, add_84, rsqrt_23, view_302, getitem_95, view_320, getitem_97, add_87, rsqrt_24, view_322, getitem_99, view_340, getitem_101, add_90, rsqrt_25, view_342, mm_86, tanh_10, mm_87, getitem_103, view_346, getitem_105, add_94, rsqrt_26, view_348, getitem_107, view_366, getitem_109, add_97, rsqrt_27, view_368, getitem_111, view_386, getitem_113, add_100, rsqrt_28, view_388, mm_97, tanh_11, mm_98, getitem_115, view_392, getitem_117, add_104, rsqrt_29, view_394, getitem_119, view_412, getitem_121, add_107, rsqrt_30, view_414, getitem_123, view_432, getitem_125, add_110, rsqrt_31, view_434, mm_108, tanh_12, mm_109, getitem_127, view_438, getitem_129, add_114, rsqrt_32, view_440, getitem_131, view_458, getitem_133, add_117, rsqrt_33, view_460, getitem_135, view_478, getitem_137, add_120, rsqrt_34, view_480, mm_119, tanh_13, mm_120, getitem_139, view_484, getitem_141, add_124, rsqrt_35, view_486, getitem_143, view_504, getitem_145, add_127, rsqrt_36, view_506, getitem_147, view_524, getitem_149, add_130, rsqrt_37, view_526, mm_130, tanh_14, mm_131, getitem_151, view_530, getitem_153, add_134, rsqrt_38, view_532, getitem_155, view_550, getitem_157, add_137, rsqrt_39, view_552, getitem_159, view_570, getitem_161, add_140, rsqrt_40, view_572, mm_141, tanh_15, mm_142, getitem_163, view_576, getitem_165, add_144, rsqrt_41, getitem_167, view_578, sub_30, convert_element_type_7, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, alias_87, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, alias_89, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, alias_93, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, alias_95, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, alias_99, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, alias_101, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, alias_105, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, alias_107, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, alias_111, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, alias_113, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, alias_117, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, alias_119, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, alias_123, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, alias_125, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, alias_129, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, alias_131, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, alias_136, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, alias_140, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, alias_144, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, alias_148, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, alias_152, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, alias_156, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, alias_160, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, alias_164, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
