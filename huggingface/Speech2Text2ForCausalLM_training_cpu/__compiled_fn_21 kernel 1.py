
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


cpp_fused_native_dropout_backward_native_layer_norm_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                auto tmp11 = out_ptr1[static_cast<long>(x0)];
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = static_cast<float>(256.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 * tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp14 = tmp9 - tmp13;
                auto tmp15 = at::vec::Vectorized<float>(tmp0);
                auto tmp16 = tmp15 * tmp14;
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr2[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr5[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_sum_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            tmp_acc1_vec = tmp_acc1_vec + tmp6;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x0)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp9 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp13 = out_ptr2[static_cast<long>(x0)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = static_cast<float>(256.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp12 * tmp14;
                        auto tmp16 = tmp11 - tmp15;
                        auto tmp17 = at::vec::Vectorized<float>(tmp0);
                        auto tmp18 = tmp17 * tmp16;
                        tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            tmp_acc1_vec = tmp_acc1_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr3[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    out_ptr6[static_cast<long>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp1 * tmp4;
                auto tmp6 = tmp2 - tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<long>(x2 + (64L*x0) + (256L*x1)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (128L*x2) + (8192L*x0)), static_cast<long>(128L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (8192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x0) + (256L*x1) + (256L*x1_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_sum_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(0.125);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_9, primals_15, view, view_16, getitem_1, mul_1, view_18, view_20, getitem_5, mul_3, div_1, permute_11, le, permute_15, div_2, permute_19, permute_24, permute_25, alias_3, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_9, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(view, (128, 256), (256, 1))
    assert_size_stride(view_16, (128, 256), (256, 1))
    assert_size_stride(getitem_1, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(mul_1, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(view_18, (128, 256), (256, 1))
    assert_size_stride(view_20, (128, 2048), (2048, 1))
    assert_size_stride(getitem_5, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(mul_3, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(div_1, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_11, (256, 2048), (2048, 1))
    assert_size_stride(le, (1, 128, 2048), (262144, 2048, 1))
    assert_size_stride(permute_15, (2048, 256), (256, 1))
    assert_size_stride(div_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_19, (256, 256), (256, 1))
    assert_size_stride(permute_24, (4, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_25, (4, 64, 128), (8192, 1, 64))
    assert_size_stride(alias_3, (4, 128, 128), (16384, 128, 1))
    assert_size_stride(permute_26, (4, 64, 128), (8192, 1, 64))
    assert_size_stride(permute_27, (4, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_31, (256, 256), (256, 1))
    assert_size_stride(permute_36, (256, 256), (256, 1))
    assert_size_stride(permute_40, (256, 256), (256, 1))
    assert_size_stride(tangents_1, (1, 128, 256), (32768, 256, 1))
    assert_size_stride(tangents_2, (1, 4, 128, 64), (32768, 8192, 64, 1))
    assert_size_stride(tangents_3, (1, 4, 128, 64), (32768, 8192, 64, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf2 = empty((1, 128, 256), device='cpu', dtype=torch.float32)
    buf3 = empty((256, ), device='cpu', dtype=torch.float32)
    buf4 = empty((256, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 128, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del div_1
    del getitem_5
    del mul_3
    del primals_15
    del tangents_1
    buf6 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (128, 256), (256, 1), 0), permute_11, out=buf6)
    del permute_11
    buf7 = empty((256, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (256, 128), (1, 256), 0), view_20, out=buf7)
    del view_20
    buf8 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf6, (1, 128, 2048), (262144, 2048, 1), 0); del buf6  # reuse
    cpp_fused_sum_threshold_backward_1(c_void_p(buf9.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf8.data_ptr()))
    del le
    buf10 = reinterpret_tensor(buf5, (128, 256), (256, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (128, 2048), (2048, 1), 0), permute_15, out=buf10)
    del permute_15
    buf11 = empty((2048, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (2048, 128), (1, 2048), 0), view_18, out=buf11)
    del view_18
    buf12 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf13 = buf1; del buf1  # reuse
    buf14 = buf0; del buf0  # reuse
    buf15 = empty((1, 128, 256), device='cpu', dtype=torch.float32)
    buf16 = empty((256, ), device='cpu', dtype=torch.float32)
    buf17 = empty((256, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1, 128, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_2(c_void_p(buf9.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del buf13
    del buf14
    del buf9
    del div_2
    del getitem_1
    del mul_1
    del primals_9
    buf19 = reinterpret_tensor(buf2, (128, 256), (256, 1), 0); del buf2  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (128, 256), (256, 1), 0), permute_19, out=buf19)
    del permute_19
    buf20 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (256, 128), (1, 256), 0), view_16, out=buf20)
    del view_16
    buf21 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_3(c_void_p(buf18.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf18, (4, 128, 64), (8192, 64, 1), 0); del buf18  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_24, reinterpret_tensor(buf19, (4, 128, 64), (64, 256, 1), 0), out=buf22)
    del permute_24
    buf23 = empty((4, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf19, (4, 128, 64), (64, 256, 1), 0), permute_25, out=buf23)
    del permute_25
    buf24 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf25 = buf23; del buf23  # reuse
    cpp_fused__softmax_backward_data_4(c_void_p(buf25.data_ptr()), c_void_p(alias_3.data_ptr()), c_void_p(buf24.data_ptr()))
    del alias_3
    del buf24
    buf26 = reinterpret_tensor(buf19, (4, 64, 128), (8192, 128, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_26, reinterpret_tensor(buf25, (4, 128, 128), (16384, 128, 1), 0), out=buf26)
    del permute_26
    buf27 = reinterpret_tensor(buf10, (4, 128, 64), (8192, 64, 1), 0); del buf10  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (4, 128, 128), (16384, 128, 1), 0), permute_27, out=buf27)
    del permute_27
    buf28 = empty((1, 128, 4, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_5(c_void_p(tangents_3.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf28.data_ptr()))
    del tangents_3
    buf29 = reinterpret_tensor(buf22, (128, 256), (256, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (128, 256), (256, 1), 0), permute_31, out=buf29)
    del permute_31
    buf30 = reinterpret_tensor(buf25, (256, 256), (256, 1), 0); del buf25  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (256, 128), (1, 256), 0), view, out=buf30)
    buf31 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 128, 4, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_6(c_void_p(buf28.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    del tangents_2
    buf33 = reinterpret_tensor(buf28, (128, 256), (256, 1), 0); del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (128, 256), (256, 1), 0), permute_36, out=buf33)
    del permute_36
    buf34 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (256, 128), (1, 256), 0), view, out=buf34)
    buf35 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf26, (128, 256), (256, 1), 0); del buf26  # reuse
    cpp_fused_mul_sum_view_7(c_void_p(buf32.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del buf27
    buf37 = reinterpret_tensor(buf32, (128, 256), (256, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf36, permute_40, out=buf37)
    del permute_40
    buf38 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (256, 128), (1, 256), 0), view, out=buf38)
    del view
    buf39 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf40 = buf15; del buf15  # reuse
    cpp_fused_add_sum_8(c_void_p(buf40.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()))
    return (reinterpret_tensor(buf38, (256, 256), (256, 1), 0), reinterpret_tensor(buf39, (256, ), (1, ), 0), reinterpret_tensor(buf34, (256, 256), (256, 1), 0), reinterpret_tensor(buf35, (256, ), (1, ), 0), reinterpret_tensor(buf30, (256, 256), (256, 1), 0), reinterpret_tensor(buf31, (256, ), (1, ), 0), reinterpret_tensor(buf20, (256, 256), (256, 1), 0), reinterpret_tensor(buf21, (256, ), (1, ), 0), buf16, buf17, reinterpret_tensor(buf11, (2048, 256), (256, 1), 0), reinterpret_tensor(buf12, (2048, ), (1, ), 0), reinterpret_tensor(buf7, (256, 2048), (2048, 1), 0), reinterpret_tensor(buf8, (256, ), (1, ), 0), buf3, buf4, buf40, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((1, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.bool)
    mul_1 = rand_strided((1, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((128, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((1, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((1, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_11 = rand_strided((256, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cpu', dtype=torch.bool)
    permute_15 = rand_strided((2048, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_19 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_24 = rand_strided((4, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_25 = rand_strided((4, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    alias_3 = rand_strided((4, 128, 128), (16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_26 = rand_strided((4, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    permute_27 = rand_strided((4, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    permute_31 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_36 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_40 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 4, 128, 64), (32768, 8192, 64, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 4, 128, 64), (32768, 8192, 64, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_15, view, view_16, getitem_1, mul_1, view_18, view_20, getitem_5, mul_3, div_1, permute_11, le, permute_15, div_2, permute_19, permute_24, permute_25, alias_3, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('Speech2Text2ForCausalLM', benchmark_compiled_module)
