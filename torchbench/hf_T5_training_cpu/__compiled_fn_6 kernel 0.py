
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


cpp_fused_add_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
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
                tmp8.store(out_ptr0 + static_cast<long>(x0));
                tmp16.store(out_ptr1 + static_cast<long>(x0));
                tmp24.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x0));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x0));
                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x0));
                auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x0));
                auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x0));
                auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x0));
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
                tmp8.store(out_ptr3 + static_cast<long>(x0));
                tmp16.store(out_ptr4 + static_cast<long>(x0));
                tmp24.store(out_ptr5 + static_cast<long>(x0));
                tmp32.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = static_cast<float>(0.04419417382415922);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = tmp5 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp6 = in_ptr5[static_cast<long>(x0)];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<float>(0.04419417382415922);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = static_cast<float>(-0.5);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp6)(tmp6 * tmp6);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp6);
                    auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                    auto tmp15 = static_cast<float>(512.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp19 = tmp17 + tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp22 = static_cast<float>(2.0);
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = at::vec::Vectorized<float>(tmp16);
                    auto tmp26 = tmp25 * tmp24;
                    auto tmp27 = tmp8 + tmp26;
                    tmp27.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = in_ptr4[static_cast<long>(x1)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp14 = in_ptr6[static_cast<long>(x1)];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp19 = in_ptr8[static_cast<long>(x1)];
                        auto tmp1 = static_cast<float>(0.04419417382415922);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp3 * tmp11;
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp6 * tmp15;
                        auto tmp17 = tmp13 * tmp16;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp4 * tmp20;
                        auto tmp22 = tmp18 * tmp21;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                        tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        tmp_acc2_vec = tmp_acc2_vec + tmp22;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_16 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp6 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    auto tmp27 = static_cast<float>(2.0);
                    auto tmp28 = at::vec::Vectorized<float>(tmp27);
                    auto tmp29 = tmp26 * tmp28;
                    auto tmp30 = at::vec::Vectorized<float>(tmp19);
                    auto tmp31 = tmp30 * tmp29;
                    auto tmp32 = tmp11 + tmp31;
                    tmp32.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp2 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp15);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp7 + tmp25;
                    tmp26.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp12 = in_ptr7[static_cast<long>(x1)];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp31 = in_ptr15[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp4 * tmp14;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp9 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp7 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 + tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp32 = at::vec::Vectorized<float>(tmp31);
                        auto tmp33 = tmp5 * tmp32;
                        auto tmp34 = tmp30 * tmp33;
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp34;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp15);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp7 + tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp2 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp15);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp7 + tmp25;
                    tmp26.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr7[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp19);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp11 + tmp27;
                    tmp28.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_mul_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = in_ptr5[static_cast<long>(x1)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = in_ptr7[static_cast<long>(x1)];
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = in_ptr13[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp0 * tmp10;
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp5 * tmp14;
                        auto tmp16 = tmp12 * tmp15;
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp3 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = at::vec::Vectorized<float>(tmp27);
                        auto tmp29 = tmp1 * tmp28;
                        auto tmp30 = tmp26 * tmp29;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp16;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp30;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp15);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp7 + tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = tmp6 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp25 = static_cast<float>(2.0);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = at::vec::Vectorized<float>(tmp19);
                    auto tmp29 = tmp28 * tmp27;
                    auto tmp30 = tmp11 + tmp29;
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr4[static_cast<long>(x0)];
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
                    tmp22.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = tmp6 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp14;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr9[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    auto tmp27 = static_cast<float>(2.0);
                    auto tmp28 = at::vec::Vectorized<float>(tmp27);
                    auto tmp29 = tmp26 * tmp28;
                    auto tmp30 = at::vec::Vectorized<float>(tmp19);
                    auto tmp31 = tmp30 * tmp29;
                    auto tmp32 = tmp11 + tmp31;
                    tmp32.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp2 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp21 = static_cast<float>(2.0);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = at::vec::Vectorized<float>(tmp15);
                    auto tmp25 = tmp24 * tmp23;
                    auto tmp26 = tmp7 + tmp25;
                    tmp26.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_81 = async_compile.cpp('''
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
                       const float* in_ptr11)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
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
                tmp24.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = in_ptr5[static_cast<long>(x1)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = in_ptr13[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp0 * tmp10;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp5 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp3 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = at::vec::Vectorized<float>(tmp27);
                        auto tmp29 = tmp1 * tmp28;
                        auto tmp30 = tmp26 * tmp29;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp30;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (1024L*x3) + (65536L*x1) + (524288L*x0)), static_cast<long>(1024L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (64L*x2_inner) + (65536L*x1) + (524288L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (512L*x2_inner) + (524288L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp12 = in_ptr7[static_cast<long>(x1)];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp31 = in_ptr15[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp15 = tmp4 * tmp14;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp9 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp7 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 + tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp32 = at::vec::Vectorized<float>(tmp31);
                        auto tmp33 = tmp5 * tmp32;
                        auto tmp34 = tmp30 * tmp33;
                        tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp34;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_sum_threshold_backward_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = in_ptr3[static_cast<long>(x0)];
                auto tmp7 = in_ptr4[static_cast<long>(x0)];
                auto tmp9 = in_ptr5[static_cast<long>(x0)];
                auto tmp11 = in_ptr0[static_cast<long>(8388608L + x0)];
                auto tmp12 = in_ptr1[static_cast<long>(8388608L + x0)];
                auto tmp14 = in_ptr2[static_cast<long>(8388608L + x0)];
                auto tmp16 = in_ptr3[static_cast<long>(8388608L + x0)];
                auto tmp18 = in_ptr4[static_cast<long>(8388608L + x0)];
                auto tmp20 = in_ptr5[static_cast<long>(8388608L + x0)];
                auto tmp23 = in_ptr0[static_cast<long>(16777216L + x0)];
                auto tmp24 = in_ptr1[static_cast<long>(16777216L + x0)];
                auto tmp26 = in_ptr2[static_cast<long>(16777216L + x0)];
                auto tmp28 = in_ptr3[static_cast<long>(16777216L + x0)];
                auto tmp30 = in_ptr4[static_cast<long>(16777216L + x0)];
                auto tmp32 = in_ptr5[static_cast<long>(16777216L + x0)];
                auto tmp35 = in_ptr0[static_cast<long>(25165824L + x0)];
                auto tmp36 = in_ptr1[static_cast<long>(25165824L + x0)];
                auto tmp38 = in_ptr2[static_cast<long>(25165824L + x0)];
                auto tmp40 = in_ptr3[static_cast<long>(25165824L + x0)];
                auto tmp42 = in_ptr4[static_cast<long>(25165824L + x0)];
                auto tmp44 = in_ptr5[static_cast<long>(25165824L + x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp21 = decltype(tmp19)(tmp19 + tmp20);
                auto tmp22 = decltype(tmp10)(tmp10 + tmp21);
                auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                auto tmp34 = decltype(tmp22)(tmp22 + tmp33);
                auto tmp37 = decltype(tmp35)(tmp35 + tmp36);
                auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                auto tmp45 = decltype(tmp43)(tmp43 + tmp44);
                auto tmp46 = decltype(tmp34)(tmp34 + tmp45);
                auto tmp47 = static_cast<bool>(0);
                auto tmp48 = static_cast<float>(0.0);
                auto tmp49 = tmp47 ? tmp48 : tmp46;
                in_out_ptr0[static_cast<long>(x0)] = tmp49;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp11 = in_ptr6[static_cast<long>(x0)];
                    auto tmp15 = out_ptr0[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<int>(-1);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 + tmp13;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                    auto tmp18 = decltype(tmp11)(tmp11 * tmp11);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    auto tmp21 = static_cast<float>(512.0);
                    auto tmp22 = tmp20 / tmp21;
                    auto tmp24 = static_cast<float>(2.0);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = at::vec::Vectorized<float>(tmp22);
                    auto tmp28 = tmp27 * tmp26;
                    auto tmp29 = tmp14 + tmp28;
                    auto tmp30 = static_cast<float>(0.0);
                    auto tmp31 = to_float_mask(tmp2);
                    auto tmp32 = at::vec::Vectorized<float>(tmp30);
                    auto tmp33 = decltype(tmp32)::blendv(tmp29, tmp32, tmp31);
                    tmp33.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16449536L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = in_ptr2[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_threshold_backward_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp15);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp7 + tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = tmp6 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp25 = static_cast<float>(2.0);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = at::vec::Vectorized<float>(tmp19);
                    auto tmp29 = tmp28 * tmp27;
                    auto tmp30 = tmp11 + tmp29;
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = in_ptr5[static_cast<long>(x1)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp31 = in_ptr15[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp0 * tmp10;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp5 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp3 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 + tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp32 = at::vec::Vectorized<float>(tmp31);
                        auto tmp33 = tmp1 * tmp32;
                        auto tmp34 = tmp30 * tmp33;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp34;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_threshold_backward_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp15);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp7 + tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = tmp6 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp25 = static_cast<float>(2.0);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = at::vec::Vectorized<float>(tmp19);
                    auto tmp29 = tmp28 * tmp27;
                    auto tmp30 = tmp11 + tmp29;
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = in_ptr5[static_cast<long>(x1)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp31 = in_ptr15[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp0 * tmp10;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp5 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp3 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 + tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp32 = at::vec::Vectorized<float>(tmp31);
                        auto tmp33 = tmp1 * tmp32;
                        auto tmp34 = tmp30 * tmp33;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp34;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    }
}
''')


cpp_fused_threshold_backward_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp23 = static_cast<float>(2.0);
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp26 = at::vec::Vectorized<float>(tmp15);
                    auto tmp27 = tmp26 * tmp25;
                    auto tmp28 = tmp7 + tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_pow_sum_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = tmp6 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp22 = tmp20 + tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp25 = static_cast<float>(2.0);
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = at::vec::Vectorized<float>(tmp19);
                    auto tmp29 = tmp28 * tmp27;
                    auto tmp30 = tmp11 + tmp29;
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_threshold_backward_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_div_mul_pow_sum_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = static_cast<float>(2.0);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp15);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp7 + tmp23;
                    tmp24.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (524288L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (65536L*x1) + (524288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_as_strided_scatter_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(33554432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((1024L*x1) + (1024L*x1_inner) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>(x0) % static_cast<long>(1024L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(1024L))) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (524288L*(c10::div_floor_integer(x0, 1024L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr15,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp8 = in_ptr5[static_cast<long>(x1)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp17 = in_ptr9[static_cast<long>(x1)];
                        auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp22 = in_ptr11[static_cast<long>(x1)];
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp31 = in_ptr15[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp0 * tmp10;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp5 * tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp3 * tmp23;
                        auto tmp25 = tmp21 * tmp24;
                        auto tmp28 = tmp26 + tmp27;
                        auto tmp30 = tmp28 + tmp29;
                        auto tmp32 = at::vec::Vectorized<float>(tmp31);
                        auto tmp33 = tmp1 * tmp32;
                        auto tmp34 = tmp30 * tmp33;
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                        tmp_acc2_vec = tmp_acc2_vec + tmp25;
                        tmp_acc3_vec = tmp_acc3_vec + tmp34;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_sum_threshold_backward_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp3 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = in_ptr3[static_cast<long>(x0)];
                auto tmp7 = in_ptr4[static_cast<long>(x0)];
                auto tmp9 = in_ptr5[static_cast<long>(x0)];
                auto tmp11 = in_ptr0[static_cast<long>(8388608L + x0)];
                auto tmp12 = in_ptr1[static_cast<long>(8388608L + x0)];
                auto tmp14 = in_ptr2[static_cast<long>(8388608L + x0)];
                auto tmp16 = in_ptr3[static_cast<long>(8388608L + x0)];
                auto tmp18 = in_ptr4[static_cast<long>(8388608L + x0)];
                auto tmp20 = in_ptr5[static_cast<long>(8388608L + x0)];
                auto tmp23 = in_ptr0[static_cast<long>(16777216L + x0)];
                auto tmp24 = in_ptr1[static_cast<long>(16777216L + x0)];
                auto tmp26 = in_ptr2[static_cast<long>(16777216L + x0)];
                auto tmp28 = in_ptr3[static_cast<long>(16777216L + x0)];
                auto tmp30 = in_ptr4[static_cast<long>(16777216L + x0)];
                auto tmp32 = in_ptr5[static_cast<long>(16777216L + x0)];
                auto tmp35 = in_ptr0[static_cast<long>(25165824L + x0)];
                auto tmp36 = in_ptr1[static_cast<long>(25165824L + x0)];
                auto tmp38 = in_ptr2[static_cast<long>(25165824L + x0)];
                auto tmp40 = in_ptr3[static_cast<long>(25165824L + x0)];
                auto tmp42 = in_ptr4[static_cast<long>(25165824L + x0)];
                auto tmp44 = in_ptr5[static_cast<long>(25165824L + x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp21 = decltype(tmp19)(tmp19 + tmp20);
                auto tmp22 = decltype(tmp10)(tmp10 + tmp21);
                auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                auto tmp34 = decltype(tmp22)(tmp22 + tmp33);
                auto tmp37 = decltype(tmp35)(tmp35 + tmp36);
                auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                auto tmp45 = decltype(tmp43)(tmp43 + tmp44);
                auto tmp46 = decltype(tmp34)(tmp34 + tmp45);
                auto tmp47 = static_cast<bool>(0);
                auto tmp48 = static_cast<float>(0.0);
                auto tmp49 = tmp47 ? tmp48 : tmp46;
                in_out_ptr0[static_cast<long>(x0)] = tmp49;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp11 = in_ptr6[static_cast<long>(x0)];
                    auto tmp15 = out_ptr0[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = static_cast<int>(-1);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp3 + tmp13;
                    auto tmp16 = static_cast<float>(-0.5);
                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                    auto tmp18 = decltype(tmp11)(tmp11 * tmp11);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    auto tmp21 = static_cast<float>(512.0);
                    auto tmp22 = tmp20 / tmp21;
                    auto tmp24 = static_cast<float>(2.0);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = at::vec::Vectorized<float>(tmp22);
                    auto tmp28 = tmp27 * tmp26;
                    auto tmp29 = tmp14 + tmp28;
                    auto tmp30 = static_cast<float>(0.0);
                    auto tmp31 = to_float_mask(tmp2);
                    auto tmp32 = at::vec::Vectorized<float>(tmp30);
                    auto tmp33 = decltype(tmp32)::blendv(tmp29, tmp32, tmp31);
                    tmp33.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16449536L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_146 = async_compile.cpp('''
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, view, embedding, rsqrt, view_1, add_3, view_19, mm_3, rsqrt_1, view_21, view_23, mm_5, rsqrt_2, view_25, view_43, mm_9, rsqrt_3, view_45, view_47, mm_11, rsqrt_4, view_49, view_67, mm_15, rsqrt_5, view_69, view_71, mm_17, rsqrt_6, view_73, view_91, mm_21, rsqrt_7, view_93, view_95, mm_23, rsqrt_8, view_97, view_115, mm_27, rsqrt_9, view_117, view_119, mm_29, rsqrt_10, view_121, view_139, mm_33, rsqrt_11, view_141, view_143, mm_35, rsqrt_12, view_145, embedding_2, rsqrt_13, view_146, add_37, view_164, mm_39, rsqrt_14, view_166, view_169, view_184, mm_43, rsqrt_15, view_186, view_188, mm_45, rsqrt_16, view_190, view_208, mm_49, rsqrt_17, view_210, view_228, mm_53, rsqrt_18, view_230, view_232, mm_55, rsqrt_19, view_234, view_252, mm_59, rsqrt_20, view_254, view_272, mm_63, rsqrt_21, view_274, view_276, mm_65, rsqrt_22, view_278, view_296, mm_69, rsqrt_23, view_298, view_316, mm_73, rsqrt_24, view_318, view_320, mm_75, rsqrt_25, view_322, view_340, mm_79, rsqrt_26, view_342, view_360, mm_83, rsqrt_27, view_362, view_364, mm_85, rsqrt_28, view_366, view_384, mm_89, rsqrt_29, view_386, view_404, mm_93, rsqrt_30, view_406, view_408, mm_95, rsqrt_31, view_410, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_65, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_67, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_71, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_73, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_77, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_79, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_83, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_85, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_89, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_91, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_95, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_97, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_102, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_106, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_110, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_114, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_118, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_122, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
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
    assert_size_stride(view, (4, 1024), (1024, 1))
    assert_size_stride(embedding, (4, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_1, (4096, 512), (512, 1))
    assert_size_stride(add_3, (1024, 1024), (1024, 1))
    assert_size_stride(view_19, (4096, 512), (512, 1))
    assert_size_stride(mm_3, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_1, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_21, (4096, 512), (512, 1))
    assert_size_stride(view_23, (4096, 2048), (2048, 1))
    assert_size_stride(mm_5, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_2, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_25, (4096, 512), (512, 1))
    assert_size_stride(view_43, (4096, 512), (512, 1))
    assert_size_stride(mm_9, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_3, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_45, (4096, 512), (512, 1))
    assert_size_stride(view_47, (4096, 2048), (2048, 1))
    assert_size_stride(mm_11, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_4, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_49, (4096, 512), (512, 1))
    assert_size_stride(view_67, (4096, 512), (512, 1))
    assert_size_stride(mm_15, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_5, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_69, (4096, 512), (512, 1))
    assert_size_stride(view_71, (4096, 2048), (2048, 1))
    assert_size_stride(mm_17, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_6, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_73, (4096, 512), (512, 1))
    assert_size_stride(view_91, (4096, 512), (512, 1))
    assert_size_stride(mm_21, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_7, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_93, (4096, 512), (512, 1))
    assert_size_stride(view_95, (4096, 2048), (2048, 1))
    assert_size_stride(mm_23, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_8, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_97, (4096, 512), (512, 1))
    assert_size_stride(view_115, (4096, 512), (512, 1))
    assert_size_stride(mm_27, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_9, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_117, (4096, 512), (512, 1))
    assert_size_stride(view_119, (4096, 2048), (2048, 1))
    assert_size_stride(mm_29, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_10, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_121, (4096, 512), (512, 1))
    assert_size_stride(view_139, (4096, 512), (512, 1))
    assert_size_stride(mm_33, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_11, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_141, (4096, 512), (512, 1))
    assert_size_stride(view_143, (4096, 2048), (2048, 1))
    assert_size_stride(mm_35, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_12, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_145, (4, 1024), (1024, 1))
    assert_size_stride(embedding_2, (4, 1024, 512), (524288, 512, 1))
    assert_size_stride(rsqrt_13, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_146, (4096, 512), (512, 1))
    assert_size_stride(add_37, (1024, 1024), (1024, 1))
    assert_size_stride(view_164, (4096, 512), (512, 1))
    assert_size_stride(mm_39, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_14, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_166, (4096, 512), (512, 1))
    assert_size_stride(view_169, (4096, 512), (512, 1))
    assert_size_stride(view_184, (4096, 512), (512, 1))
    assert_size_stride(mm_43, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_15, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_186, (4096, 512), (512, 1))
    assert_size_stride(view_188, (4096, 2048), (2048, 1))
    assert_size_stride(mm_45, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_16, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_190, (4096, 512), (512, 1))
    assert_size_stride(view_208, (4096, 512), (512, 1))
    assert_size_stride(mm_49, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_17, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_210, (4096, 512), (512, 1))
    assert_size_stride(view_228, (4096, 512), (512, 1))
    assert_size_stride(mm_53, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_18, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_230, (4096, 512), (512, 1))
    assert_size_stride(view_232, (4096, 2048), (2048, 1))
    assert_size_stride(mm_55, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_19, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_234, (4096, 512), (512, 1))
    assert_size_stride(view_252, (4096, 512), (512, 1))
    assert_size_stride(mm_59, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_20, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_254, (4096, 512), (512, 1))
    assert_size_stride(view_272, (4096, 512), (512, 1))
    assert_size_stride(mm_63, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_21, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_274, (4096, 512), (512, 1))
    assert_size_stride(view_276, (4096, 2048), (2048, 1))
    assert_size_stride(mm_65, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_22, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_278, (4096, 512), (512, 1))
    assert_size_stride(view_296, (4096, 512), (512, 1))
    assert_size_stride(mm_69, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_23, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_298, (4096, 512), (512, 1))
    assert_size_stride(view_316, (4096, 512), (512, 1))
    assert_size_stride(mm_73, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_24, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_318, (4096, 512), (512, 1))
    assert_size_stride(view_320, (4096, 2048), (2048, 1))
    assert_size_stride(mm_75, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_25, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_322, (4096, 512), (512, 1))
    assert_size_stride(view_340, (4096, 512), (512, 1))
    assert_size_stride(mm_79, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_26, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_342, (4096, 512), (512, 1))
    assert_size_stride(view_360, (4096, 512), (512, 1))
    assert_size_stride(mm_83, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_27, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_362, (4096, 512), (512, 1))
    assert_size_stride(view_364, (4096, 2048), (2048, 1))
    assert_size_stride(mm_85, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_28, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_366, (4096, 512), (512, 1))
    assert_size_stride(view_384, (4096, 512), (512, 1))
    assert_size_stride(mm_89, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_29, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_386, (4096, 512), (512, 1))
    assert_size_stride(view_404, (4096, 512), (512, 1))
    assert_size_stride(mm_93, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_30, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_406, (4096, 512), (512, 1))
    assert_size_stride(view_408, (4096, 2048), (2048, 1))
    assert_size_stride(mm_95, (4096, 512), (512, 1))
    assert_size_stride(rsqrt_31, (4, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_410, (4096, 512), (512, 1))
    assert_size_stride(permute_191, (32128, 512), (512, 1))
    assert_size_stride(permute_195, (512, 2048), (2048, 1))
    assert_size_stride(le_1, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_199, (2048, 512), (512, 1))
    assert_size_stride(permute_203, (512, 512), (512, 1))
    assert_size_stride(permute_206, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_207, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_65, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_208, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_209, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_214, (512, 512), (512, 1))
    assert_size_stride(permute_219, (512, 512), (512, 1))
    assert_size_stride(permute_224, (512, 512), (512, 1))
    assert_size_stride(permute_228, (512, 512), (512, 1))
    assert_size_stride(permute_231, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_232, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_67, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_233, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_234, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_239, (512, 512), (512, 1))
    assert_size_stride(permute_244, (512, 512), (512, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(permute_253, (512, 2048), (2048, 1))
    assert_size_stride(le_2, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_257, (2048, 512), (512, 1))
    assert_size_stride(permute_261, (512, 512), (512, 1))
    assert_size_stride(permute_264, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_265, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_71, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_266, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_267, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_272, (512, 512), (512, 1))
    assert_size_stride(permute_277, (512, 512), (512, 1))
    assert_size_stride(permute_282, (512, 512), (512, 1))
    assert_size_stride(permute_286, (512, 512), (512, 1))
    assert_size_stride(permute_289, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_290, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_73, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_291, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_292, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_297, (512, 512), (512, 1))
    assert_size_stride(permute_302, (512, 512), (512, 1))
    assert_size_stride(permute_307, (512, 512), (512, 1))
    assert_size_stride(permute_311, (512, 2048), (2048, 1))
    assert_size_stride(le_3, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_315, (2048, 512), (512, 1))
    assert_size_stride(permute_319, (512, 512), (512, 1))
    assert_size_stride(permute_322, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_323, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_77, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_324, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_325, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_330, (512, 512), (512, 1))
    assert_size_stride(permute_335, (512, 512), (512, 1))
    assert_size_stride(permute_340, (512, 512), (512, 1))
    assert_size_stride(permute_344, (512, 512), (512, 1))
    assert_size_stride(permute_347, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_348, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_79, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_349, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_350, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_355, (512, 512), (512, 1))
    assert_size_stride(permute_360, (512, 512), (512, 1))
    assert_size_stride(permute_365, (512, 512), (512, 1))
    assert_size_stride(permute_369, (512, 2048), (2048, 1))
    assert_size_stride(le_4, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_373, (2048, 512), (512, 1))
    assert_size_stride(permute_377, (512, 512), (512, 1))
    assert_size_stride(permute_380, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_381, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_83, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_382, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_383, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_388, (512, 512), (512, 1))
    assert_size_stride(permute_393, (512, 512), (512, 1))
    assert_size_stride(permute_398, (512, 512), (512, 1))
    assert_size_stride(permute_402, (512, 512), (512, 1))
    assert_size_stride(permute_405, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_406, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_85, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_407, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_408, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_413, (512, 512), (512, 1))
    assert_size_stride(permute_418, (512, 512), (512, 1))
    assert_size_stride(permute_423, (512, 512), (512, 1))
    assert_size_stride(permute_427, (512, 2048), (2048, 1))
    assert_size_stride(le_5, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_431, (2048, 512), (512, 1))
    assert_size_stride(permute_435, (512, 512), (512, 1))
    assert_size_stride(permute_438, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_439, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_89, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_440, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_441, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_446, (512, 512), (512, 1))
    assert_size_stride(permute_451, (512, 512), (512, 1))
    assert_size_stride(permute_456, (512, 512), (512, 1))
    assert_size_stride(permute_460, (512, 512), (512, 1))
    assert_size_stride(permute_463, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_464, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_91, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_465, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_466, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_471, (512, 512), (512, 1))
    assert_size_stride(permute_476, (512, 512), (512, 1))
    assert_size_stride(permute_481, (512, 512), (512, 1))
    assert_size_stride(permute_485, (512, 2048), (2048, 1))
    assert_size_stride(le_6, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_489, (2048, 512), (512, 1))
    assert_size_stride(permute_493, (512, 512), (512, 1))
    assert_size_stride(permute_496, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_497, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_95, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_498, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_499, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_504, (512, 512), (512, 1))
    assert_size_stride(permute_509, (512, 512), (512, 1))
    assert_size_stride(permute_514, (512, 512), (512, 1))
    assert_size_stride(permute_518, (512, 512), (512, 1))
    assert_size_stride(permute_521, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_522, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_97, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_524, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_525, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_530, (512, 512), (512, 1))
    assert_size_stride(permute_535, (512, 512), (512, 1))
    assert_size_stride(permute_540, (512, 512), (512, 1))
    assert_size_stride(permute_544, (512, 2048), (2048, 1))
    assert_size_stride(le_7, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_548, (2048, 512), (512, 1))
    assert_size_stride(permute_552, (512, 512), (512, 1))
    assert_size_stride(permute_555, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_556, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_102, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_557, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_558, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_563, (512, 512), (512, 1))
    assert_size_stride(permute_568, (512, 512), (512, 1))
    assert_size_stride(permute_573, (512, 512), (512, 1))
    assert_size_stride(permute_577, (512, 2048), (2048, 1))
    assert_size_stride(le_8, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_581, (2048, 512), (512, 1))
    assert_size_stride(permute_585, (512, 512), (512, 1))
    assert_size_stride(permute_588, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_589, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_106, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_590, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_591, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_596, (512, 512), (512, 1))
    assert_size_stride(permute_601, (512, 512), (512, 1))
    assert_size_stride(permute_606, (512, 512), (512, 1))
    assert_size_stride(permute_610, (512, 2048), (2048, 1))
    assert_size_stride(le_9, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_614, (2048, 512), (512, 1))
    assert_size_stride(permute_618, (512, 512), (512, 1))
    assert_size_stride(permute_621, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_622, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_110, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_623, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_624, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_629, (512, 512), (512, 1))
    assert_size_stride(permute_634, (512, 512), (512, 1))
    assert_size_stride(permute_639, (512, 512), (512, 1))
    assert_size_stride(permute_643, (512, 2048), (2048, 1))
    assert_size_stride(le_10, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_647, (2048, 512), (512, 1))
    assert_size_stride(permute_651, (512, 512), (512, 1))
    assert_size_stride(permute_654, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_655, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_114, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_656, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_657, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_662, (512, 512), (512, 1))
    assert_size_stride(permute_667, (512, 512), (512, 1))
    assert_size_stride(permute_672, (512, 512), (512, 1))
    assert_size_stride(permute_676, (512, 2048), (2048, 1))
    assert_size_stride(le_11, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_680, (2048, 512), (512, 1))
    assert_size_stride(permute_684, (512, 512), (512, 1))
    assert_size_stride(permute_687, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_688, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_118, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_689, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_690, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_695, (512, 512), (512, 1))
    assert_size_stride(permute_700, (512, 512), (512, 1))
    assert_size_stride(permute_705, (512, 512), (512, 1))
    assert_size_stride(permute_709, (512, 2048), (2048, 1))
    assert_size_stride(le_12, (4, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(permute_713, (2048, 512), (512, 1))
    assert_size_stride(permute_717, (512, 512), (512, 1))
    assert_size_stride(permute_720, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_721, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_122, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1))
    assert_size_stride(permute_723, (32, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_724, (32, 1024, 64), (65536, 1, 1024))
    assert_size_stride(permute_729, (512, 512), (512, 1))
    assert_size_stride(permute_734, (512, 512), (512, 1))
    assert_size_stride(permute_739, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (4, 1024, 32128), (32899072, 32128, 1))
    assert_size_stride(tangents_2, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_3, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_4, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_5, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_6, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_7, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_8, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_9, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_10, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_11, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_12, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_13, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_14, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_15, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_16, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_17, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_18, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_19, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_20, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_21, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_22, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_23, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_24, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_25, (4, 8, 1024, 64), (524288, 65536, 64, 1))
    assert_size_stride(tangents_26, (4, 1024, 512), (524288, 512, 1))
    buf0 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf1 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf2 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf5 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    buf6 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_0(c_void_p(embedding.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(mm_11.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(mm_21.data_ptr()), c_void_p(mm_23.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(mm_33.data_ptr()), c_void_p(mm_35.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(mm_43.data_ptr()), c_void_p(mm_45.data_ptr()), c_void_p(mm_49.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(mm_55.data_ptr()), c_void_p(mm_59.data_ptr()), c_void_p(mm_63.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(mm_69.data_ptr()), c_void_p(mm_73.data_ptr()), c_void_p(mm_75.data_ptr()), c_void_p(mm_79.data_ptr()), c_void_p(mm_83.data_ptr()), c_void_p(mm_85.data_ptr()), c_void_p(mm_89.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del mm_11
    del mm_23
    del mm_35
    del mm_49
    del mm_63
    del mm_75
    del mm_89
    buf7 = empty((32128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (32128, 4096), (1, 32128), 0), view_410, out=buf7)
    del view_410
    buf8 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (4096, 32128), (32128, 1), 0), permute_191, out=buf8)
    del permute_191
    del tangents_1
    buf10 = empty_strided((4, 1024, 1), (1024, 1, 4096), device='cpu', dtype=torch.float32)
    buf11 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_1(c_void_p(buf8.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(mm_93.data_ptr()), c_void_p(mm_95.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_32
    buf13 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (4096, 512), (512, 1), 0), permute_195, out=buf13)
    del permute_195
    buf14 = reinterpret_tensor(buf13, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf13  # reuse
    cpp_fused_threshold_backward_2(c_void_p(buf14.data_ptr()), c_void_p(le_1.data_ptr()))
    del le_1
    buf16 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (4096, 2048), (2048, 1), 0), permute_199, out=buf16)
    del permute_199
    buf18 = buf10; del buf10  # reuse
    buf19 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_3(c_void_p(buf16.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(mm_93.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_31
    buf21 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (4096, 512), (512, 1), 0), permute_203, out=buf21)
    del permute_203
    buf22 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_4(c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf24 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (32, 1024, 64), (65536, 64, 1), 0), permute_207, out=buf24)
    del permute_207
    buf25 = empty_strided((4, 8, 1024, 1), (8192, 1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf24, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf24  # reuse
    buf28 = empty((33554432, ), device='cpu', dtype=torch.float32)
    buf31 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_5(c_void_p(buf26.data_ptr()), c_void_p(alias_65.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf31.data_ptr()))
    del alias_65
    buf34 = reinterpret_tensor(buf21, (32, 1024, 64), (65536, 64, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf31, permute_209, out=buf34)
    del permute_209
    buf41 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf34.data_ptr()), c_void_p(buf41.data_ptr()))
    buf43 = reinterpret_tensor(buf34, (4096, 512), (512, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf41, permute_224, out=buf43)
    del permute_224
    buf9 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf44 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_7(c_void_p(buf8.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(mm_93.data_ptr()), c_void_p(mm_95.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf44.data_ptr()))
    del mm_93
    del mm_95
    del rsqrt_30
    del rsqrt_31
    buf12 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (512, 4096), (1, 512), 0), view_408, out=buf12)
    del view_408
    buf15 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (2048, 4096), (1, 2048), 0), view_406, out=buf15)
    del view_406
    buf20 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (512, 4096), (1, 512), 0), view_404, out=buf20)
    del view_404
    buf23 = reinterpret_tensor(buf11, (32, 1024, 64), (65536, 64, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_206, reinterpret_tensor(buf22, (32, 1024, 64), (65536, 64, 1), 0), out=buf23)
    del permute_206
    buf33 = reinterpret_tensor(buf22, (32, 64, 1024), (65536, 1024, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_208, buf31, out=buf33)
    del permute_208
    buf35 = reinterpret_tensor(buf8, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf8  # reuse
    cpp_fused_clone_8(c_void_p(tangents_25.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf35.data_ptr()))
    del tangents_25
    buf36 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (512, 4096), (1, 512), 0), view_169, out=buf36)
    buf37 = reinterpret_tensor(buf23, (4096, 512), (512, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (4096, 512), (512, 1), 0), permute_214, out=buf37)
    del permute_214
    buf38 = buf35; del buf35  # reuse
    cpp_fused_clone_9(c_void_p(tangents_24.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf38.data_ptr()))
    del tangents_24
    buf39 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (512, 4096), (1, 512), 0), view_169, out=buf39)
    buf40 = reinterpret_tensor(buf33, (4096, 512), (512, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (4096, 512), (512, 1), 0), permute_219, out=buf40)
    del permute_219
    buf42 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (512, 4096), (1, 512), 0), view_386, out=buf42)
    del view_386
    buf45 = buf18; del buf18  # reuse
    buf46 = buf19; del buf19  # reuse
    cpp_fused_add_div_mul_pow_sum_10(c_void_p(buf46.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(buf45.data_ptr()))
    del primals_30
    del rsqrt_29
    buf47 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (512, 4096), (1, 512), 0), view_384, out=buf47)
    del view_384
    buf48 = reinterpret_tensor(buf6, (4096, 512), (512, 1), 0); del buf6  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (4096, 512), (512, 1), 0), permute_228, out=buf48)
    del permute_228
    buf49 = reinterpret_tensor(buf43, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf43  # reuse
    cpp_fused_clone_11(c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    buf50 = reinterpret_tensor(buf48, (32, 1024, 64), (65536, 64, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_231, reinterpret_tensor(buf49, (32, 1024, 64), (65536, 64, 1), 0), out=buf50)
    del permute_231
    buf51 = buf31; del buf31  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (32, 1024, 64), (65536, 64, 1), 0), permute_232, out=buf51)
    del permute_232
    buf52 = buf25; del buf25  # reuse
    buf53 = reinterpret_tensor(buf51, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf51  # reuse
    buf54 = buf28; del buf28  # reuse
    buf57 = reinterpret_tensor(buf26, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf26  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_12(c_void_p(buf53.data_ptr()), c_void_p(alias_67.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf57.data_ptr()))
    del alias_67
    buf59 = reinterpret_tensor(buf49, (32, 64, 1024), (65536, 1024, 1), 0); del buf49  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_233, buf57, out=buf59)
    del permute_233
    buf60 = reinterpret_tensor(buf41, (32, 1024, 64), (65536, 64, 1), 0); del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf57, permute_234, out=buf60)
    del permute_234
    buf61 = buf38; del buf38  # reuse
    cpp_fused_clone_13(c_void_p(tangents_23.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf61.data_ptr()))
    del tangents_23
    buf62 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (512, 4096), (1, 512), 0), view_366, out=buf62)
    buf63 = reinterpret_tensor(buf50, (4096, 512), (512, 1), 0); del buf50  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (4096, 512), (512, 1), 0), permute_239, out=buf63)
    del permute_239
    buf64 = buf61; del buf61  # reuse
    cpp_fused_clone_14(c_void_p(tangents_22.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf64.data_ptr()))
    del tangents_22
    buf65 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (512, 4096), (1, 512), 0), view_366, out=buf65)
    buf66 = reinterpret_tensor(buf59, (4096, 512), (512, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (4096, 512), (512, 1), 0), permute_244, out=buf66)
    del permute_244
    buf67 = reinterpret_tensor(buf64, (4096, 512), (512, 1), 0); del buf64  # reuse
    cpp_fused_view_15(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (512, 4096), (1, 512), 0), view_366, out=buf68)
    del view_366
    buf69 = reinterpret_tensor(buf60, (4096, 512), (512, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf67, permute_249, out=buf69)
    del permute_249
    buf71 = buf45; del buf45  # reuse
    buf72 = buf46; del buf46  # reuse
    cpp_fused_add_div_mul_pow_sum_16(c_void_p(buf72.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(mm_79.data_ptr()), c_void_p(mm_83.data_ptr()), c_void_p(mm_85.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_29
    buf74 = reinterpret_tensor(buf14, (4096, 2048), (2048, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (4096, 512), (512, 1), 0), permute_253, out=buf74)
    del permute_253
    buf75 = reinterpret_tensor(buf74, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf74  # reuse
    cpp_fused_threshold_backward_17(c_void_p(buf75.data_ptr()), c_void_p(le_2.data_ptr()))
    del le_2
    buf77 = buf67; del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (4096, 2048), (2048, 1), 0), permute_257, out=buf77)
    del permute_257
    buf79 = buf71; del buf71  # reuse
    buf80 = reinterpret_tensor(buf16, (4, 1024, 512), (524288, 512, 1), 0); del buf16  # reuse
    cpp_fused_add_div_mul_pow_sum_18(c_void_p(buf77.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(mm_79.data_ptr()), c_void_p(mm_83.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del primals_28
    buf82 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (4096, 512), (512, 1), 0), permute_261, out=buf82)
    del permute_261
    buf83 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_19(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    buf85 = buf57; del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf83, (32, 1024, 64), (65536, 64, 1), 0), permute_265, out=buf85)
    del permute_265
    buf86 = buf52; del buf52  # reuse
    buf87 = reinterpret_tensor(buf85, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf85  # reuse
    buf88 = reinterpret_tensor(buf53, (33554432, ), (1, ), 0); del buf53  # reuse
    buf91 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_20(c_void_p(buf87.data_ptr()), c_void_p(alias_71.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf91.data_ptr()))
    del alias_71
    buf94 = reinterpret_tensor(buf82, (32, 1024, 64), (65536, 64, 1), 0); del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf91, permute_267, out=buf94)
    del permute_267
    buf101 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_21(c_void_p(buf94.data_ptr()), c_void_p(buf101.data_ptr()))
    buf103 = reinterpret_tensor(buf94, (4096, 512), (512, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf101, permute_282, out=buf103)
    del permute_282
    buf105 = buf79; del buf79  # reuse
    buf106 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_22(c_void_p(buf103.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(mm_79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_27
    buf108 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (4096, 512), (512, 1), 0), permute_286, out=buf108)
    del permute_286
    buf109 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_23(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf108, (32, 1024, 64), (65536, 64, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_289, reinterpret_tensor(buf109, (32, 1024, 64), (65536, 64, 1), 0), out=buf110)
    del permute_289
    buf121 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_24(c_void_p(tangents_19.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf121.data_ptr()))
    del tangents_19
    buf123 = reinterpret_tensor(buf110, (4096, 512), (512, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (4096, 512), (512, 1), 0), permute_297, out=buf123)
    del permute_297
    buf111 = reinterpret_tensor(buf88, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (32, 1024, 64), (65536, 64, 1), 0), permute_290, out=buf111)
    del permute_290
    buf112 = buf86; del buf86  # reuse
    buf113 = reinterpret_tensor(buf111, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf111  # reuse
    buf114 = reinterpret_tensor(buf87, (33554432, ), (1, ), 0); del buf87  # reuse
    buf117 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_25(c_void_p(buf113.data_ptr()), c_void_p(alias_73.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf117.data_ptr()))
    del alias_73
    buf119 = reinterpret_tensor(buf109, (32, 64, 1024), (65536, 1024, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_291, buf117, out=buf119)
    del permute_291
    buf124 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_26(c_void_p(tangents_18.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf124.data_ptr()))
    del tangents_18
    buf126 = reinterpret_tensor(buf119, (4096, 512), (512, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (4096, 512), (512, 1), 0), permute_302, out=buf126)
    del permute_302
    buf120 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf117, permute_292, out=buf120)
    del permute_292
    buf127 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_27(c_void_p(buf120.data_ptr()), c_void_p(buf127.data_ptr()))
    buf129 = reinterpret_tensor(buf120, (4096, 512), (512, 1), 0); del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf127, permute_307, out=buf129)
    del permute_307
    buf70 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf78 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf104 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf130 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_28(c_void_p(buf63.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(mm_79.data_ptr()), c_void_p(mm_83.data_ptr()), c_void_p(mm_85.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf130.data_ptr()))
    del mm_79
    del mm_83
    del mm_85
    del rsqrt_26
    del rsqrt_27
    del rsqrt_28
    buf73 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (512, 4096), (1, 512), 0), view_364, out=buf73)
    del view_364
    buf76 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (2048, 4096), (1, 2048), 0), view_362, out=buf76)
    del view_362
    buf81 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (512, 4096), (1, 512), 0), view_360, out=buf81)
    del view_360
    buf84 = reinterpret_tensor(buf80, (32, 1024, 64), (65536, 64, 1), 0); del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_264, reinterpret_tensor(buf83, (32, 1024, 64), (65536, 64, 1), 0), out=buf84)
    del permute_264
    buf93 = reinterpret_tensor(buf83, (32, 64, 1024), (65536, 1024, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_266, buf91, out=buf93)
    del permute_266
    buf95 = reinterpret_tensor(buf72, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf72  # reuse
    cpp_fused_clone_29(c_void_p(tangents_21.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf95.data_ptr()))
    del tangents_21
    buf96 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (512, 4096), (1, 512), 0), view_169, out=buf96)
    buf97 = reinterpret_tensor(buf84, (4096, 512), (512, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (4096, 512), (512, 1), 0), permute_272, out=buf97)
    del permute_272
    buf98 = buf95; del buf95  # reuse
    cpp_fused_clone_30(c_void_p(tangents_20.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf98.data_ptr()))
    del tangents_20
    buf99 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (512, 4096), (1, 512), 0), view_169, out=buf99)
    buf100 = reinterpret_tensor(buf93, (4096, 512), (512, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (4096, 512), (512, 1), 0), permute_277, out=buf100)
    del permute_277
    buf102 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (512, 4096), (1, 512), 0), view_342, out=buf102)
    del view_342
    buf107 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (512, 4096), (1, 512), 0), view_340, out=buf107)
    del view_340
    buf122 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (512, 4096), (1, 512), 0), view_322, out=buf122)
    buf125 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (512, 4096), (1, 512), 0), view_322, out=buf125)
    buf128 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (512, 4096), (1, 512), 0), view_322, out=buf128)
    del view_322
    buf131 = buf105; del buf105  # reuse
    buf132 = buf106; del buf106  # reuse
    cpp_fused_add_div_mul_pow_sum_31(c_void_p(buf132.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_26
    del rsqrt_25
    buf133 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (512, 4096), (1, 512), 0), view_320, out=buf133)
    del view_320
    buf134 = reinterpret_tensor(buf75, (4096, 2048), (2048, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (4096, 512), (512, 1), 0), permute_311, out=buf134)
    del permute_311
    buf135 = reinterpret_tensor(buf134, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf134  # reuse
    cpp_fused_threshold_backward_32(c_void_p(buf135.data_ptr()), c_void_p(le_3.data_ptr()))
    del le_3
    buf136 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (2048, 4096), (1, 2048), 0), view_318, out=buf136)
    del view_318
    buf137 = reinterpret_tensor(buf5, (4096, 512), (512, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (4096, 2048), (2048, 1), 0), permute_315, out=buf137)
    del permute_315
    buf139 = buf131; del buf131  # reuse
    buf140 = buf132; del buf132  # reuse
    cpp_fused_add_div_mul_pow_sum_33(c_void_p(buf140.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(mm_69.data_ptr()), c_void_p(mm_73.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_25
    buf142 = buf129; del buf129  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (4096, 512), (512, 1), 0), permute_319, out=buf142)
    del permute_319
    buf143 = reinterpret_tensor(buf126, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf126  # reuse
    cpp_fused_clone_34(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf145 = buf91; del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf143, (32, 1024, 64), (65536, 64, 1), 0), permute_323, out=buf145)
    del permute_323
    buf146 = buf112; del buf112  # reuse
    buf147 = reinterpret_tensor(buf145, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf145  # reuse
    buf148 = reinterpret_tensor(buf117, (33554432, ), (1, ), 0); del buf117  # reuse
    buf151 = reinterpret_tensor(buf113, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf113  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_35(c_void_p(buf147.data_ptr()), c_void_p(alias_77.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf151.data_ptr()))
    del alias_77
    buf154 = reinterpret_tensor(buf142, (32, 1024, 64), (65536, 64, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf151, permute_325, out=buf154)
    del permute_325
    buf161 = buf123; del buf123  # reuse
    cpp_fused_view_36(c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()))
    buf163 = reinterpret_tensor(buf154, (4096, 512), (512, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf161, permute_340, out=buf163)
    del permute_340
    buf165 = buf139; del buf139  # reuse
    buf166 = reinterpret_tensor(buf127, (4, 1024, 512), (524288, 512, 1), 0); del buf127  # reuse
    cpp_fused_add_div_mul_pow_sum_37(c_void_p(buf163.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(mm_69.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_24
    buf168 = reinterpret_tensor(buf124, (4096, 512), (512, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (4096, 512), (512, 1), 0), permute_344, out=buf168)
    del permute_344
    buf169 = reinterpret_tensor(buf121, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf121  # reuse
    cpp_fused_clone_38(c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    buf170 = reinterpret_tensor(buf168, (32, 1024, 64), (65536, 64, 1), 0); del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_347, reinterpret_tensor(buf169, (32, 1024, 64), (65536, 64, 1), 0), out=buf170)
    del permute_347
    buf181 = reinterpret_tensor(buf101, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf101  # reuse
    cpp_fused_clone_39(c_void_p(tangents_15.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf181.data_ptr()))
    del tangents_15
    buf183 = reinterpret_tensor(buf170, (4096, 512), (512, 1), 0); del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (4096, 512), (512, 1), 0), permute_355, out=buf183)
    del permute_355
    buf171 = reinterpret_tensor(buf148, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf148  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (32, 1024, 64), (65536, 64, 1), 0), permute_348, out=buf171)
    del permute_348
    buf172 = buf146; del buf146  # reuse
    buf173 = reinterpret_tensor(buf171, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf171  # reuse
    buf174 = reinterpret_tensor(buf147, (33554432, ), (1, ), 0); del buf147  # reuse
    buf177 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_40(c_void_p(buf173.data_ptr()), c_void_p(alias_79.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf177.data_ptr()))
    del alias_79
    buf179 = reinterpret_tensor(buf169, (32, 64, 1024), (65536, 1024, 1), 0); del buf169  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_349, buf177, out=buf179)
    del permute_349
    buf184 = buf98; del buf98  # reuse
    cpp_fused_clone_41(c_void_p(tangents_14.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf184.data_ptr()))
    del tangents_14
    buf186 = reinterpret_tensor(buf179, (4096, 512), (512, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (4096, 512), (512, 1), 0), permute_360, out=buf186)
    del permute_360
    buf180 = reinterpret_tensor(buf77, (32, 1024, 64), (65536, 64, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf177, permute_350, out=buf180)
    del permute_350
    buf187 = buf69; del buf69  # reuse
    cpp_fused_view_42(c_void_p(buf180.data_ptr()), c_void_p(buf187.data_ptr()))
    buf189 = reinterpret_tensor(buf180, (4096, 512), (512, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, permute_365, out=buf189)
    del permute_365
    buf191 = buf165; del buf165  # reuse
    buf192 = reinterpret_tensor(buf66, (4, 1024, 512), (524288, 512, 1), 0); del buf66  # reuse
    cpp_fused_add_div_mul_pow_sum_43(c_void_p(buf183.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del primals_23
    buf194 = reinterpret_tensor(buf135, (4096, 2048), (2048, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (4096, 512), (512, 1), 0), permute_369, out=buf194)
    del permute_369
    buf195 = reinterpret_tensor(buf194, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf194  # reuse
    cpp_fused_threshold_backward_44(c_void_p(buf195.data_ptr()), c_void_p(le_4.data_ptr()))
    del le_4
    buf197 = buf63; del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (4096, 2048), (2048, 1), 0), permute_373, out=buf197)
    del permute_373
    buf138 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf164 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf190 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf198 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_45(c_void_p(buf137.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(mm_65.data_ptr()), c_void_p(mm_69.data_ptr()), c_void_p(mm_73.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf198.data_ptr()))
    del mm_65
    del mm_69
    del mm_73
    del rsqrt_22
    del rsqrt_23
    del rsqrt_24
    buf141 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (512, 4096), (1, 512), 0), view_316, out=buf141)
    del view_316
    buf144 = reinterpret_tensor(buf140, (32, 1024, 64), (65536, 64, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_322, reinterpret_tensor(buf143, (32, 1024, 64), (65536, 64, 1), 0), out=buf144)
    del permute_322
    buf153 = reinterpret_tensor(buf143, (32, 64, 1024), (65536, 1024, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_324, buf151, out=buf153)
    del permute_324
    buf155 = reinterpret_tensor(buf189, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf189  # reuse
    cpp_fused_clone_46(c_void_p(tangents_17.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf155.data_ptr()))
    del tangents_17
    buf156 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (512, 4096), (1, 512), 0), view_169, out=buf156)
    buf157 = reinterpret_tensor(buf144, (4096, 512), (512, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (4096, 512), (512, 1), 0), permute_330, out=buf157)
    del permute_330
    buf158 = buf155; del buf155  # reuse
    cpp_fused_clone_47(c_void_p(tangents_16.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf158.data_ptr()))
    del tangents_16
    buf159 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (512, 4096), (1, 512), 0), view_169, out=buf159)
    buf160 = reinterpret_tensor(buf153, (4096, 512), (512, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (4096, 512), (512, 1), 0), permute_335, out=buf160)
    del permute_335
    buf162 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (512, 4096), (1, 512), 0), view_298, out=buf162)
    del view_298
    buf167 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (1, 512), 0), view_296, out=buf167)
    del view_296
    buf182 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (512, 4096), (1, 512), 0), view_278, out=buf182)
    buf185 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (512, 4096), (1, 512), 0), view_278, out=buf185)
    buf188 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (512, 4096), (1, 512), 0), view_278, out=buf188)
    del view_278
    buf193 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (512, 4096), (1, 512), 0), view_276, out=buf193)
    del view_276
    buf196 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (2048, 4096), (1, 2048), 0), view_274, out=buf196)
    del view_274
    buf199 = buf191; del buf191  # reuse
    buf200 = buf192; del buf192  # reuse
    cpp_fused_add_div_mul_pow_sum_48(c_void_p(buf200.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_22
    del rsqrt_21
    buf201 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf200, (512, 4096), (1, 512), 0), view_272, out=buf201)
    del view_272
    buf202 = reinterpret_tensor(buf4, (4096, 512), (512, 1), 0); del buf4  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf200, (4096, 512), (512, 1), 0), permute_377, out=buf202)
    del permute_377
    buf203 = reinterpret_tensor(buf197, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf197  # reuse
    cpp_fused_clone_49(c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = reinterpret_tensor(buf202, (32, 1024, 64), (65536, 64, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_380, reinterpret_tensor(buf203, (32, 1024, 64), (65536, 64, 1), 0), out=buf204)
    del permute_380
    buf205 = buf151; del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf203, (32, 1024, 64), (65536, 64, 1), 0), permute_381, out=buf205)
    del permute_381
    buf206 = buf172; del buf172  # reuse
    buf207 = reinterpret_tensor(buf205, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf205  # reuse
    buf208 = reinterpret_tensor(buf177, (33554432, ), (1, ), 0); del buf177  # reuse
    buf211 = reinterpret_tensor(buf173, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf173  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_50(c_void_p(buf207.data_ptr()), c_void_p(alias_83.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf211.data_ptr()))
    del alias_83
    buf213 = reinterpret_tensor(buf203, (32, 64, 1024), (65536, 1024, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_382, buf211, out=buf213)
    del permute_382
    buf214 = reinterpret_tensor(buf187, (32, 1024, 64), (65536, 64, 1), 0); del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf211, permute_383, out=buf214)
    del permute_383
    buf215 = buf184; del buf184  # reuse
    cpp_fused_clone_51(c_void_p(tangents_13.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf215.data_ptr()))
    del tangents_13
    buf216 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (512, 4096), (1, 512), 0), view_169, out=buf216)
    buf217 = reinterpret_tensor(buf204, (4096, 512), (512, 1), 0); del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (4096, 512), (512, 1), 0), permute_388, out=buf217)
    del permute_388
    buf218 = buf215; del buf215  # reuse
    cpp_fused_clone_52(c_void_p(tangents_12.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf218.data_ptr()))
    del tangents_12
    buf219 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (512, 4096), (1, 512), 0), view_169, out=buf219)
    buf220 = reinterpret_tensor(buf213, (4096, 512), (512, 1), 0); del buf213  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (4096, 512), (512, 1), 0), permute_393, out=buf220)
    del permute_393
    buf222 = reinterpret_tensor(buf218, (4096, 512), (512, 1), 0); del buf218  # reuse
    cpp_fused_view_53(c_void_p(buf214.data_ptr()), c_void_p(buf222.data_ptr()))
    buf224 = reinterpret_tensor(buf214, (4096, 512), (512, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf222, permute_398, out=buf224)
    del permute_398
    buf226 = buf199; del buf199  # reuse
    buf227 = buf200; del buf200  # reuse
    cpp_fused_add_div_mul_pow_sum_54(c_void_p(buf227.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(mm_55.data_ptr()), c_void_p(mm_59.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf226.data_ptr()))
    del primals_21
    buf229 = reinterpret_tensor(buf181, (4096, 512), (512, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (4096, 512), (512, 1), 0), permute_402, out=buf229)
    del permute_402
    buf230 = reinterpret_tensor(buf166, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf166  # reuse
    cpp_fused_clone_55(c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf229, (32, 1024, 64), (65536, 64, 1), 0); del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_405, reinterpret_tensor(buf230, (32, 1024, 64), (65536, 64, 1), 0), out=buf231)
    del permute_405
    buf242 = reinterpret_tensor(buf161, (4, 1024, 8, 64), (524288, 512, 64, 1), 0); del buf161  # reuse
    cpp_fused_clone_56(c_void_p(tangents_11.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf242.data_ptr()))
    del tangents_11
    buf244 = reinterpret_tensor(buf231, (4096, 512), (512, 1), 0); del buf231  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (4096, 512), (512, 1), 0), permute_413, out=buf244)
    del permute_413
    buf232 = buf211; del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (32, 1024, 64), (65536, 64, 1), 0), permute_406, out=buf232)
    del permute_406
    buf233 = buf206; del buf206  # reuse
    buf234 = reinterpret_tensor(buf232, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf232  # reuse
    buf235 = buf208; del buf208  # reuse
    buf238 = reinterpret_tensor(buf207, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf207  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_57(c_void_p(buf234.data_ptr()), c_void_p(alias_85.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf238.data_ptr()))
    del alias_85
    buf240 = reinterpret_tensor(buf230, (32, 64, 1024), (65536, 1024, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_407, buf238, out=buf240)
    del permute_407
    buf245 = buf158; del buf158  # reuse
    cpp_fused_clone_58(c_void_p(tangents_10.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf245.data_ptr()))
    del tangents_10
    buf247 = reinterpret_tensor(buf240, (4096, 512), (512, 1), 0); del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (4096, 512), (512, 1), 0), permute_418, out=buf247)
    del permute_418
    buf241 = reinterpret_tensor(buf186, (32, 1024, 64), (65536, 64, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf238, permute_408, out=buf241)
    del permute_408
    buf248 = buf183; del buf183  # reuse
    cpp_fused_view_59(c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()))
    buf250 = reinterpret_tensor(buf241, (4096, 512), (512, 1), 0); del buf241  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf248, permute_423, out=buf250)
    del permute_423
    buf252 = buf226; del buf226  # reuse
    buf253 = reinterpret_tensor(buf163, (4, 1024, 512), (524288, 512, 1), 0); del buf163  # reuse
    cpp_fused_add_div_mul_pow_sum_60(c_void_p(buf244.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(mm_55.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del primals_20
    buf255 = reinterpret_tensor(buf195, (4096, 2048), (2048, 1), 0); del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf253, (4096, 512), (512, 1), 0), permute_427, out=buf255)
    del permute_427
    buf256 = reinterpret_tensor(buf255, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf255  # reuse
    cpp_fused_threshold_backward_61(c_void_p(buf256.data_ptr()), c_void_p(le_5.data_ptr()))
    del le_5
    buf258 = buf137; del buf137  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (4096, 2048), (2048, 1), 0), permute_431, out=buf258)
    del permute_431
    buf260 = buf252; del buf252  # reuse
    buf261 = reinterpret_tensor(buf103, (4, 1024, 512), (524288, 512, 1), 0); del buf103  # reuse
    cpp_fused_add_div_mul_pow_sum_62(c_void_p(buf258.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del primals_19
    buf263 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (4096, 512), (512, 1), 0), permute_435, out=buf263)
    del permute_435
    buf264 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_63(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf263, (32, 1024, 64), (65536, 64, 1), 0); del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_438, reinterpret_tensor(buf264, (32, 1024, 64), (65536, 64, 1), 0), out=buf265)
    del permute_438
    buf276 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_64(c_void_p(tangents_9.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf276.data_ptr()))
    del tangents_9
    buf278 = reinterpret_tensor(buf265, (4096, 512), (512, 1), 0); del buf265  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (4096, 512), (512, 1), 0), permute_446, out=buf278)
    del permute_446
    buf266 = buf238; del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf264, (32, 1024, 64), (65536, 64, 1), 0), permute_439, out=buf266)
    del permute_439
    buf267 = buf233; del buf233  # reuse
    buf268 = reinterpret_tensor(buf266, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf266  # reuse
    buf269 = reinterpret_tensor(buf234, (33554432, ), (1, ), 0); del buf234  # reuse
    buf272 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_65(c_void_p(buf268.data_ptr()), c_void_p(alias_89.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf272.data_ptr()))
    del alias_89
    buf274 = reinterpret_tensor(buf264, (32, 64, 1024), (65536, 1024, 1), 0); del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_440, buf272, out=buf274)
    del permute_440
    buf279 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_66(c_void_p(tangents_8.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf279.data_ptr()))
    del tangents_8
    buf281 = reinterpret_tensor(buf274, (4096, 512), (512, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (4096, 512), (512, 1), 0), permute_451, out=buf281)
    del permute_451
    buf275 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf272, permute_441, out=buf275)
    del permute_441
    buf282 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_67(c_void_p(buf275.data_ptr()), c_void_p(buf282.data_ptr()))
    buf284 = reinterpret_tensor(buf275, (4096, 512), (512, 1), 0); del buf275  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf282, permute_456, out=buf284)
    del permute_456
    buf286 = buf260; del buf260  # reuse
    buf287 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_68(c_void_p(buf284.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del primals_18
    buf289 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (4096, 512), (512, 1), 0), permute_460, out=buf289)
    del permute_460
    buf290 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_69(c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    buf291 = reinterpret_tensor(buf289, (32, 1024, 64), (65536, 64, 1), 0); del buf289  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_463, reinterpret_tensor(buf290, (32, 1024, 64), (65536, 64, 1), 0), out=buf291)
    del permute_463
    buf302 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(tangents_7.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf302.data_ptr()))
    del tangents_7
    buf304 = reinterpret_tensor(buf291, (4096, 512), (512, 1), 0); del buf291  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (4096, 512), (512, 1), 0), permute_471, out=buf304)
    del permute_471
    buf292 = buf272; del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf290, (32, 1024, 64), (65536, 64, 1), 0), permute_464, out=buf292)
    del permute_464
    buf293 = buf267; del buf267  # reuse
    buf294 = reinterpret_tensor(buf292, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf292  # reuse
    buf295 = buf269; del buf269  # reuse
    buf298 = reinterpret_tensor(buf268, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf268  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_71(c_void_p(buf294.data_ptr()), c_void_p(alias_91.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf298.data_ptr()))
    del alias_91
    buf300 = reinterpret_tensor(buf290, (32, 64, 1024), (65536, 1024, 1), 0); del buf290  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_465, buf298, out=buf300)
    del permute_465
    buf305 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_72(c_void_p(tangents_6.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf305.data_ptr()))
    del tangents_6
    buf307 = reinterpret_tensor(buf300, (4096, 512), (512, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (4096, 512), (512, 1), 0), permute_476, out=buf307)
    del permute_476
    buf301 = empty((32, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf298, permute_466, out=buf301)
    del permute_466
    buf308 = empty((4096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_73(c_void_p(buf301.data_ptr()), c_void_p(buf308.data_ptr()))
    buf310 = reinterpret_tensor(buf301, (4096, 512), (512, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf308, permute_481, out=buf310)
    del permute_481
    buf312 = buf286; del buf286  # reuse
    buf313 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_74(c_void_p(buf304.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(mm_43.data_ptr()), c_void_p(mm_45.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_17
    buf315 = empty((4096, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf313, (4096, 512), (512, 1), 0), permute_485, out=buf315)
    del permute_485
    buf316 = reinterpret_tensor(buf315, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf315  # reuse
    cpp_fused_threshold_backward_75(c_void_p(buf316.data_ptr()), c_void_p(le_6.data_ptr()))
    del le_6
    buf318 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf316, (4096, 2048), (2048, 1), 0), permute_489, out=buf318)
    del permute_489
    buf320 = buf312; del buf312  # reuse
    buf321 = empty((4, 1024, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_pow_sum_76(c_void_p(buf318.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(mm_43.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del primals_16
    buf323 = empty((4096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (4096, 512), (512, 1), 0), permute_493, out=buf323)
    del permute_493
    buf324 = empty((4, 8, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_77(c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()))
    buf325 = reinterpret_tensor(buf323, (32, 1024, 64), (65536, 64, 1), 0); del buf323  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_496, reinterpret_tensor(buf324, (32, 1024, 64), (65536, 64, 1), 0), out=buf325)
    del permute_496
    buf336 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_78(c_void_p(tangents_5.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf336.data_ptr()))
    del tangents_5
    buf338 = reinterpret_tensor(buf325, (4096, 512), (512, 1), 0); del buf325  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (4096, 512), (512, 1), 0), permute_504, out=buf338)
    del permute_504
    buf326 = buf298; del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf324, (32, 1024, 64), (65536, 64, 1), 0), permute_497, out=buf326)
    del permute_497
    buf327 = buf293; del buf293  # reuse
    buf328 = reinterpret_tensor(buf326, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf326  # reuse
    buf329 = reinterpret_tensor(buf294, (33554432, ), (1, ), 0); del buf294  # reuse
    buf332 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_as_strided_scatter_79(c_void_p(buf328.data_ptr()), c_void_p(alias_95.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf332.data_ptr()))
    del alias_95
    buf334 = reinterpret_tensor(buf324, (32, 64, 1024), (65536, 1024, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_498, buf332, out=buf334)
    del permute_498
    buf339 = empty((4, 1024, 8, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_80(c_void_p(tangents_4.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf339.data_ptr()))
    del tangents_4
    buf341 = reinterpret_tensor(buf334, (4096, 512), (512, 1), 0); del buf334  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (4096, 512), (512, 1), 0), permute_509, out=buf341)
    del permute_509
    buf221 = reinterpret_tensor(buf100, (4, 1024, 512), (524288, 512, 1), 0); del buf100  # reuse
    buf342 = buf221; del buf221  # reuse
    cpp_fused_add_81(c_void_p(buf342.data_ptr()), c_void_p(tangents_26.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf341.data_ptr()))
    del buf157
    del buf160
    del buf217
    del buf220
    del buf278
    del buf281
    del buf338
    del buf341
    del buf37
    del buf40
    del buf97
    del tangents_26
    buf223 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (512, 4096), (1, 512), 0), view_254, out=buf223)
    del buf222
    del view_254
    buf225 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf251 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf259 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf285 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_82(c_void_p(buf224.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(mm_53.data_ptr()), c_void_p(mm_55.data_ptr()), c_void_p(mm_59.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf285.data_ptr()))
    del buf224
    del buf244
    del buf247
    del buf250
    del buf258
    del buf284
    del buf3
    del mm_53
    del mm_55
    del mm_59
    del rsqrt_17
    del rsqrt_18
    del rsqrt_19
    del rsqrt_20
    buf228 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (512, 4096), (1, 512), 0), view_252, out=buf228)
    del buf227
    del view_252
    buf243 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (512, 4096), (1, 512), 0), view_234, out=buf243)
    del buf242
    buf246 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (512, 4096), (1, 512), 0), view_234, out=buf246)
    del buf245
    buf249 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (512, 4096), (1, 512), 0), view_234, out=buf249)
    del buf248
    del view_234
    buf254 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf253, (512, 4096), (1, 512), 0), view_232, out=buf254)
    del view_232
    buf257 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (2048, 4096), (1, 2048), 0), view_230, out=buf257)
    del buf256
    del view_230
    buf262 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (512, 4096), (1, 512), 0), view_228, out=buf262)
    del view_228
    buf277 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (512, 4096), (1, 512), 0), view_169, out=buf277)
    buf280 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (512, 4096), (1, 512), 0), view_169, out=buf280)
    buf283 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (512, 4096), (1, 512), 0), view_210, out=buf283)
    del view_210
    buf288 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (512, 4096), (1, 512), 0), view_208, out=buf288)
    del view_208
    buf303 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (512, 4096), (1, 512), 0), view_190, out=buf303)
    buf306 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (512, 4096), (1, 512), 0), view_190, out=buf306)
    buf309 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (512, 4096), (1, 512), 0), view_190, out=buf309)
    del view_190
    buf335 = reinterpret_tensor(buf308, (32, 1024, 64), (65536, 64, 1), 0); del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf332, permute_499, out=buf335)
    del permute_499
    buf343 = reinterpret_tensor(buf305, (4096, 512), (512, 1), 0); del buf305  # reuse
    cpp_fused_view_83(c_void_p(buf335.data_ptr()), c_void_p(buf343.data_ptr()))
    buf345 = reinterpret_tensor(buf335, (4096, 512), (512, 1), 0); del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf343, permute_514, out=buf345)
    del permute_514
    buf347 = buf320; del buf320  # reuse
    buf348 = reinterpret_tensor(buf302, (4, 1024, 512), (524288, 512, 1), 0); del buf302  # reuse
    cpp_fused_add_div_mul_pow_sum_84(c_void_p(buf345.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del primals_15
    buf350 = reinterpret_tensor(buf287, (4096, 512), (512, 1), 0); del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (4096, 512), (512, 1), 0), permute_518, out=buf350)
    del permute_518
    buf351 = reinterpret_tensor(buf282, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf282  # reuse
    cpp_fused_clone_85(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    buf352 = reinterpret_tensor(buf350, (32, 1024, 64), (65536, 64, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_521, reinterpret_tensor(buf351, (32, 1024, 64), (65536, 64, 1), 0), out=buf352)
    del permute_521
    buf368 = buf279; del buf279  # reuse
    cpp_fused_clone_86(c_void_p(tangents_3.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf368.data_ptr()))
    del tangents_3
    buf370 = reinterpret_tensor(buf352, (4096, 512), (512, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (4096, 512), (512, 1), 0), permute_530, out=buf370)
    del permute_530
    buf353 = buf332; del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf351, (32, 1024, 64), (65536, 64, 1), 0), permute_522, out=buf353)
    del permute_522
    buf354 = buf327; del buf327  # reuse
    buf355 = reinterpret_tensor(buf353, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf353  # reuse
    buf356 = buf329; del buf329  # reuse
    buf359 = reinterpret_tensor(buf328, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf328  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_87(c_void_p(buf355.data_ptr()), c_void_p(alias_97.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf359.data_ptr()))
    del alias_97
    buf366 = reinterpret_tensor(buf351, (32, 64, 1024), (65536, 1024, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_524, buf359, out=buf366)
    del permute_524
    buf371 = buf276; del buf276  # reuse
    cpp_fused_clone_88(c_void_p(tangents_2.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf371.data_ptr()))
    del tangents_2
    buf373 = reinterpret_tensor(buf366, (4096, 512), (512, 1), 0); del buf366  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (4096, 512), (512, 1), 0), permute_535, out=buf373)
    del permute_535
    buf367 = reinterpret_tensor(buf261, (32, 1024, 64), (65536, 64, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf359, permute_525, out=buf367)
    del permute_525
    buf374 = reinterpret_tensor(buf253, (4096, 512), (512, 1), 0); del buf253  # reuse
    cpp_fused_view_89(c_void_p(buf367.data_ptr()), c_void_p(buf374.data_ptr()))
    buf376 = reinterpret_tensor(buf367, (4096, 512), (512, 1), 0); del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf374, permute_540, out=buf376)
    del permute_540
    buf311 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf319 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf346 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf377 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_90(c_void_p(buf304.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(mm_39.data_ptr()), c_void_p(mm_43.data_ptr()), c_void_p(mm_45.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf377.data_ptr()))
    del buf304
    del buf307
    del mm_39
    del mm_43
    del mm_45
    del rsqrt_14
    del rsqrt_15
    del rsqrt_16
    buf314 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf313, (512, 4096), (1, 512), 0), view_188, out=buf314)
    del view_188
    buf317 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf316, (2048, 4096), (1, 2048), 0), view_186, out=buf317)
    del view_186
    buf322 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (512, 4096), (1, 512), 0), view_184, out=buf322)
    del view_184
    buf337 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (512, 4096), (1, 512), 0), view_169, out=buf337)
    buf340 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (512, 4096), (1, 512), 0), view_169, out=buf340)
    del view_169
    buf344 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (512, 4096), (1, 512), 0), view_166, out=buf344)
    del view_166
    buf349 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (512, 4096), (1, 512), 0), view_164, out=buf349)
    del view_164
    buf361 = reinterpret_tensor(buf316, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf316  # reuse
    buf363 = reinterpret_tensor(buf361, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf361  # reuse
    buf362 = empty((32, 8), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_sum_threshold_backward_91(c_void_p(buf363.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf362.data_ptr()))
    aten.index_put_(buf362, [add_37], buf363, True)
    del add_37
    buf369 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (512, 4096), (1, 512), 0), view_146, out=buf369)
    buf372 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (512, 4096), (1, 512), 0), view_146, out=buf372)
    buf375 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (512, 4096), (1, 512), 0), view_146, out=buf375)
    del view_146
    buf378 = buf347; del buf347  # reuse
    buf379 = buf348; del buf348  # reuse
    buf380 = empty((32128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_92(c_void_p(buf379.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(embedding_2.data_ptr()), c_void_p(view_145.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()))
    del embedding_2
    del primals_14
    del rsqrt_13
    aten.index_put_(buf380, [view_145], buf379, True)
    del view_145
    buf383 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf384 = buf378; del buf378  # reuse
    buf385 = buf2; del buf2  # reuse
    cpp_fused_add_div_mul_pow_sum_93(c_void_p(buf385.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del primals_13
    del rsqrt_12
    buf386 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (512, 4096), (1, 512), 0), view_143, out=buf386)
    del view_143
    buf387 = reinterpret_tensor(buf363, (4096, 2048), (2048, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (4096, 512), (512, 1), 0), permute_544, out=buf387)
    del permute_544
    buf388 = reinterpret_tensor(buf387, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf387  # reuse
    cpp_fused_threshold_backward_94(c_void_p(buf388.data_ptr()), c_void_p(le_7.data_ptr()))
    del le_7
    buf389 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf388, (2048, 4096), (1, 2048), 0), view_141, out=buf389)
    del view_141
    buf390 = reinterpret_tensor(buf342, (4096, 512), (512, 1), 0); del buf342  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf388, (4096, 2048), (2048, 1), 0), permute_548, out=buf390)
    del permute_548
    buf392 = buf384; del buf384  # reuse
    buf393 = buf385; del buf385  # reuse
    cpp_fused_add_div_mul_pow_sum_95(c_void_p(buf393.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(mm_33.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(buf392.data_ptr()))
    del primals_12
    buf395 = reinterpret_tensor(buf379, (4096, 512), (512, 1), 0); del buf379  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf393, (4096, 512), (512, 1), 0), permute_552, out=buf395)
    del permute_552
    buf396 = reinterpret_tensor(buf376, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf376  # reuse
    cpp_fused_clone_96(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = reinterpret_tensor(buf395, (32, 1024, 64), (65536, 64, 1), 0); del buf395  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_555, reinterpret_tensor(buf396, (32, 1024, 64), (65536, 64, 1), 0), out=buf397)
    del permute_555
    buf408 = buf373; del buf373  # reuse
    cpp_fused_view_97(c_void_p(buf397.data_ptr()), c_void_p(buf408.data_ptr()))
    buf410 = reinterpret_tensor(buf397, (4096, 512), (512, 1), 0); del buf397  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf408, permute_563, out=buf410)
    del permute_563
    buf398 = reinterpret_tensor(buf54, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf54  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf396, (32, 1024, 64), (65536, 64, 1), 0), permute_556, out=buf398)
    del permute_556
    buf399 = buf354; del buf354  # reuse
    buf400 = reinterpret_tensor(buf398, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf398  # reuse
    buf401 = buf356; del buf356  # reuse
    buf404 = reinterpret_tensor(buf295, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf295  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_98(c_void_p(buf400.data_ptr()), c_void_p(alias_102.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf404.data_ptr()))
    del alias_102
    buf406 = reinterpret_tensor(buf396, (32, 64, 1024), (65536, 1024, 1), 0); del buf396  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_557, buf404, out=buf406)
    del permute_557
    buf411 = buf370; del buf370  # reuse
    cpp_fused__unsafe_view_clone_99(c_void_p(buf406.data_ptr()), c_void_p(buf411.data_ptr()))
    buf413 = reinterpret_tensor(buf406, (4096, 512), (512, 1), 0); del buf406  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf411, permute_568, out=buf413)
    del permute_568
    buf407 = reinterpret_tensor(buf374, (32, 1024, 64), (65536, 64, 1), 0); del buf374  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf404, permute_558, out=buf407)
    del permute_558
    buf414 = reinterpret_tensor(buf371, (4096, 512), (512, 1), 0); del buf371  # reuse
    cpp_fused_view_100(c_void_p(buf407.data_ptr()), c_void_p(buf414.data_ptr()))
    buf416 = reinterpret_tensor(buf407, (4096, 512), (512, 1), 0); del buf407  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf414, permute_573, out=buf416)
    del permute_573
    buf418 = buf392; del buf392  # reuse
    buf419 = reinterpret_tensor(buf368, (4, 1024, 512), (524288, 512, 1), 0); del buf368  # reuse
    cpp_fused_add_div_mul_pow_sum_101(c_void_p(buf410.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del primals_11
    buf421 = reinterpret_tensor(buf388, (4096, 2048), (2048, 1), 0); del buf388  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (4096, 512), (512, 1), 0), permute_577, out=buf421)
    del permute_577
    buf422 = reinterpret_tensor(buf421, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf421  # reuse
    cpp_fused_threshold_backward_102(c_void_p(buf422.data_ptr()), c_void_p(le_8.data_ptr()))
    del le_8
    buf424 = buf343; del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (4096, 2048), (2048, 1), 0), permute_581, out=buf424)
    del permute_581
    buf426 = buf418; del buf418  # reuse
    buf427 = reinterpret_tensor(buf339, (4, 1024, 512), (524288, 512, 1), 0); del buf339  # reuse
    cpp_fused_add_div_mul_pow_sum_103(c_void_p(buf424.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(rsqrt_9.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_10
    buf429 = reinterpret_tensor(buf336, (4096, 512), (512, 1), 0); del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (4096, 512), (512, 1), 0), permute_585, out=buf429)
    del permute_585
    buf430 = reinterpret_tensor(buf321, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf321  # reuse
    cpp_fused_clone_104(c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    buf431 = reinterpret_tensor(buf429, (32, 1024, 64), (65536, 64, 1), 0); del buf429  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_588, reinterpret_tensor(buf430, (32, 1024, 64), (65536, 64, 1), 0), out=buf431)
    del permute_588
    buf442 = reinterpret_tensor(buf313, (4096, 512), (512, 1), 0); del buf313  # reuse
    cpp_fused_view_105(c_void_p(buf431.data_ptr()), c_void_p(buf442.data_ptr()))
    buf444 = reinterpret_tensor(buf431, (4096, 512), (512, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf442, permute_596, out=buf444)
    del permute_596
    buf432 = buf404; del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf430, (32, 1024, 64), (65536, 64, 1), 0), permute_589, out=buf432)
    del permute_589
    buf433 = buf399; del buf399  # reuse
    buf434 = reinterpret_tensor(buf432, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf432  # reuse
    buf435 = reinterpret_tensor(buf400, (33554432, ), (1, ), 0); del buf400  # reuse
    buf438 = reinterpret_tensor(buf235, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf235  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_106(c_void_p(buf434.data_ptr()), c_void_p(alias_106.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf438.data_ptr()))
    del alias_106
    buf440 = reinterpret_tensor(buf430, (32, 64, 1024), (65536, 1024, 1), 0); del buf430  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_590, buf438, out=buf440)
    del permute_590
    buf445 = buf345; del buf345  # reuse
    cpp_fused__unsafe_view_clone_107(c_void_p(buf440.data_ptr()), c_void_p(buf445.data_ptr()))
    buf447 = reinterpret_tensor(buf440, (4096, 512), (512, 1), 0); del buf440  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf445, permute_601, out=buf447)
    del permute_601
    buf441 = reinterpret_tensor(buf318, (32, 1024, 64), (65536, 64, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf438, permute_591, out=buf441)
    del permute_591
    buf448 = buf310; del buf310  # reuse
    cpp_fused_view_108(c_void_p(buf441.data_ptr()), c_void_p(buf448.data_ptr()))
    buf450 = reinterpret_tensor(buf441, (4096, 512), (512, 1), 0); del buf441  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf448, permute_606, out=buf450)
    del permute_606
    buf391 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf417 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf425 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf451 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_109(c_void_p(buf390.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(mm_27.data_ptr()), c_void_p(mm_29.data_ptr()), c_void_p(mm_33.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(rsqrt_9.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(rsqrt_8.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf451.data_ptr()))
    del buf390
    del mm_27
    del mm_29
    del mm_33
    del rsqrt_10
    del rsqrt_11
    del rsqrt_9
    buf394 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf393, (512, 4096), (1, 512), 0), view_139, out=buf394)
    del view_139
    buf409 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (512, 4096), (1, 512), 0), view_121, out=buf409)
    buf412 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf411, (512, 4096), (1, 512), 0), view_121, out=buf412)
    buf415 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf414, (512, 4096), (1, 512), 0), view_121, out=buf415)
    del view_121
    buf420 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (512, 4096), (1, 512), 0), view_119, out=buf420)
    del view_119
    buf423 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (2048, 4096), (1, 2048), 0), view_117, out=buf423)
    del view_117
    buf428 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (512, 4096), (1, 512), 0), view_115, out=buf428)
    del view_115
    buf443 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (512, 4096), (1, 512), 0), view_97, out=buf443)
    buf446 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (512, 4096), (1, 512), 0), view_97, out=buf446)
    buf449 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (512, 4096), (1, 512), 0), view_97, out=buf449)
    del view_97
    buf452 = buf426; del buf426  # reuse
    buf453 = buf1; del buf1  # reuse
    cpp_fused_add_div_mul_pow_sum_110(c_void_p(buf453.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(rsqrt_8.data_ptr()), c_void_p(buf452.data_ptr()))
    del primals_9
    del rsqrt_8
    buf454 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (512, 4096), (1, 512), 0), view_95, out=buf454)
    del view_95
    buf455 = reinterpret_tensor(buf422, (4096, 2048), (2048, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (4096, 512), (512, 1), 0), permute_610, out=buf455)
    del permute_610
    buf456 = reinterpret_tensor(buf455, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf455  # reuse
    cpp_fused_threshold_backward_111(c_void_p(buf456.data_ptr()), c_void_p(le_9.data_ptr()))
    del le_9
    buf457 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf456, (2048, 4096), (1, 2048), 0), view_93, out=buf457)
    del view_93
    buf458 = buf450; del buf450  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf456, (4096, 2048), (2048, 1), 0), permute_614, out=buf458)
    del permute_614
    buf460 = buf452; del buf452  # reuse
    buf461 = buf453; del buf453  # reuse
    cpp_fused_add_div_mul_pow_sum_112(c_void_p(buf461.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(mm_21.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(buf460.data_ptr()))
    del primals_8
    buf463 = buf447; del buf447  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (4096, 512), (512, 1), 0), permute_618, out=buf463)
    del permute_618
    buf464 = reinterpret_tensor(buf444, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf444  # reuse
    cpp_fused_clone_113(c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()))
    buf465 = reinterpret_tensor(buf463, (32, 1024, 64), (65536, 64, 1), 0); del buf463  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_621, reinterpret_tensor(buf464, (32, 1024, 64), (65536, 64, 1), 0), out=buf465)
    del permute_621
    buf476 = reinterpret_tensor(buf427, (4096, 512), (512, 1), 0); del buf427  # reuse
    cpp_fused_view_114(c_void_p(buf465.data_ptr()), c_void_p(buf476.data_ptr()))
    buf478 = reinterpret_tensor(buf465, (4096, 512), (512, 1), 0); del buf465  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf476, permute_629, out=buf478)
    del permute_629
    buf466 = buf438; del buf438  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf464, (32, 1024, 64), (65536, 64, 1), 0), permute_622, out=buf466)
    del permute_622
    buf467 = buf433; del buf433  # reuse
    buf468 = reinterpret_tensor(buf466, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf466  # reuse
    buf469 = reinterpret_tensor(buf434, (33554432, ), (1, ), 0); del buf434  # reuse
    buf472 = reinterpret_tensor(buf174, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf174  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_115(c_void_p(buf468.data_ptr()), c_void_p(alias_110.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf472.data_ptr()))
    del alias_110
    buf474 = reinterpret_tensor(buf464, (32, 64, 1024), (65536, 1024, 1), 0); del buf464  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_623, buf472, out=buf474)
    del permute_623
    buf479 = buf448; del buf448  # reuse
    cpp_fused__unsafe_view_clone_116(c_void_p(buf474.data_ptr()), c_void_p(buf479.data_ptr()))
    buf481 = reinterpret_tensor(buf474, (4096, 512), (512, 1), 0); del buf474  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf479, permute_634, out=buf481)
    del permute_634
    buf475 = reinterpret_tensor(buf445, (32, 1024, 64), (65536, 64, 1), 0); del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf472, permute_624, out=buf475)
    del permute_624
    buf482 = buf442; del buf442  # reuse
    cpp_fused_view_117(c_void_p(buf475.data_ptr()), c_void_p(buf482.data_ptr()))
    buf484 = reinterpret_tensor(buf475, (4096, 512), (512, 1), 0); del buf475  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf482, permute_639, out=buf484)
    del permute_639
    buf486 = buf460; del buf460  # reuse
    buf487 = buf419; del buf419  # reuse
    cpp_fused_add_div_mul_pow_sum_118(c_void_p(buf478.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    del primals_7
    buf489 = reinterpret_tensor(buf456, (4096, 2048), (2048, 1), 0); del buf456  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf487, (4096, 512), (512, 1), 0), permute_643, out=buf489)
    del permute_643
    buf490 = reinterpret_tensor(buf489, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf489  # reuse
    cpp_fused_threshold_backward_119(c_void_p(buf490.data_ptr()), c_void_p(le_10.data_ptr()))
    del le_10
    buf492 = buf414; del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf490, (4096, 2048), (2048, 1), 0), permute_647, out=buf492)
    del permute_647
    buf494 = buf486; del buf486  # reuse
    buf495 = reinterpret_tensor(buf411, (4, 1024, 512), (524288, 512, 1), 0); del buf411  # reuse
    cpp_fused_add_div_mul_pow_sum_120(c_void_p(buf492.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(rsqrt_5.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    del primals_6
    buf497 = buf408; del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (4096, 512), (512, 1), 0), permute_651, out=buf497)
    del permute_651
    buf498 = reinterpret_tensor(buf393, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf393  # reuse
    cpp_fused_clone_121(c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    buf499 = reinterpret_tensor(buf497, (32, 1024, 64), (65536, 64, 1), 0); del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_654, reinterpret_tensor(buf498, (32, 1024, 64), (65536, 64, 1), 0), out=buf499)
    del permute_654
    buf510 = buf424; del buf424  # reuse
    cpp_fused_view_122(c_void_p(buf499.data_ptr()), c_void_p(buf510.data_ptr()))
    buf512 = reinterpret_tensor(buf499, (4096, 512), (512, 1), 0); del buf499  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf510, permute_662, out=buf512)
    del permute_662
    buf500 = buf472; del buf472  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf498, (32, 1024, 64), (65536, 64, 1), 0), permute_655, out=buf500)
    del permute_655
    buf501 = buf467; del buf467  # reuse
    buf502 = reinterpret_tensor(buf500, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf500  # reuse
    buf503 = reinterpret_tensor(buf468, (33554432, ), (1, ), 0); del buf468  # reuse
    buf506 = reinterpret_tensor(buf114, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf114  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_123(c_void_p(buf502.data_ptr()), c_void_p(alias_114.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf506.data_ptr()))
    del alias_114
    buf508 = reinterpret_tensor(buf498, (32, 64, 1024), (65536, 1024, 1), 0); del buf498  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_656, buf506, out=buf508)
    del permute_656
    buf513 = buf416; del buf416  # reuse
    cpp_fused__unsafe_view_clone_124(c_void_p(buf508.data_ptr()), c_void_p(buf513.data_ptr()))
    buf515 = reinterpret_tensor(buf508, (4096, 512), (512, 1), 0); del buf508  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf513, permute_667, out=buf515)
    del permute_667
    buf509 = reinterpret_tensor(buf413, (32, 1024, 64), (65536, 64, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf506, permute_657, out=buf509)
    del permute_657
    buf516 = buf410; del buf410  # reuse
    cpp_fused_view_125(c_void_p(buf509.data_ptr()), c_void_p(buf516.data_ptr()))
    buf518 = reinterpret_tensor(buf509, (4096, 512), (512, 1), 0); del buf509  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf516, permute_672, out=buf518)
    del permute_672
    buf459 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf485 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf493 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf519 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_126(c_void_p(buf458.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mm_15.data_ptr()), c_void_p(mm_17.data_ptr()), c_void_p(mm_21.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(rsqrt_5.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(rsqrt_4.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf519.data_ptr()))
    del buf458
    del mm_15
    del mm_17
    del mm_21
    del rsqrt_5
    del rsqrt_6
    del rsqrt_7
    buf462 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (512, 4096), (1, 512), 0), view_91, out=buf462)
    del view_91
    buf477 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (512, 4096), (1, 512), 0), view_73, out=buf477)
    buf480 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf479, (512, 4096), (1, 512), 0), view_73, out=buf480)
    buf483 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (512, 4096), (1, 512), 0), view_73, out=buf483)
    del view_73
    buf488 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf487, (512, 4096), (1, 512), 0), view_71, out=buf488)
    del view_71
    buf491 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf490, (2048, 4096), (1, 2048), 0), view_69, out=buf491)
    del view_69
    buf496 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (512, 4096), (1, 512), 0), view_67, out=buf496)
    del view_67
    buf511 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf510, (512, 4096), (1, 512), 0), view_49, out=buf511)
    buf514 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf513, (512, 4096), (1, 512), 0), view_49, out=buf514)
    buf517 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf516, (512, 4096), (1, 512), 0), view_49, out=buf517)
    del view_49
    buf520 = buf494; del buf494  # reuse
    buf521 = buf0; del buf0  # reuse
    cpp_fused_add_div_mul_pow_sum_127(c_void_p(buf521.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(rsqrt_4.data_ptr()), c_void_p(buf520.data_ptr()))
    del primals_5
    del rsqrt_4
    buf522 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (512, 4096), (1, 512), 0), view_47, out=buf522)
    del view_47
    buf523 = reinterpret_tensor(buf490, (4096, 2048), (2048, 1), 0); del buf490  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (4096, 512), (512, 1), 0), permute_676, out=buf523)
    del permute_676
    buf524 = reinterpret_tensor(buf523, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf523  # reuse
    cpp_fused_threshold_backward_128(c_void_p(buf524.data_ptr()), c_void_p(le_11.data_ptr()))
    del le_11
    buf525 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (2048, 4096), (1, 2048), 0), view_45, out=buf525)
    del view_45
    buf526 = buf518; del buf518  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (4096, 2048), (2048, 1), 0), permute_680, out=buf526)
    del permute_680
    buf528 = buf520; del buf520  # reuse
    buf529 = buf521; del buf521  # reuse
    cpp_fused_add_div_mul_pow_sum_129(c_void_p(buf529.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(embedding.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(buf528.data_ptr()))
    del primals_4
    buf531 = buf515; del buf515  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (4096, 512), (512, 1), 0), permute_684, out=buf531)
    del permute_684
    buf532 = reinterpret_tensor(buf512, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf512  # reuse
    cpp_fused_clone_130(c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = reinterpret_tensor(buf531, (32, 1024, 64), (65536, 64, 1), 0); del buf531  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_687, reinterpret_tensor(buf532, (32, 1024, 64), (65536, 64, 1), 0), out=buf533)
    del permute_687
    buf544 = reinterpret_tensor(buf495, (4096, 512), (512, 1), 0); del buf495  # reuse
    cpp_fused_view_131(c_void_p(buf533.data_ptr()), c_void_p(buf544.data_ptr()))
    buf546 = reinterpret_tensor(buf533, (4096, 512), (512, 1), 0); del buf533  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf544, permute_695, out=buf546)
    del permute_695
    buf534 = buf506; del buf506  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf532, (32, 1024, 64), (65536, 64, 1), 0), permute_688, out=buf534)
    del permute_688
    buf535 = buf501; del buf501  # reuse
    buf536 = reinterpret_tensor(buf534, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf534  # reuse
    buf537 = reinterpret_tensor(buf502, (33554432, ), (1, ), 0); del buf502  # reuse
    buf540 = buf359; del buf359  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_132(c_void_p(buf536.data_ptr()), c_void_p(alias_118.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf540.data_ptr()))
    del alias_118
    buf542 = reinterpret_tensor(buf532, (32, 64, 1024), (65536, 1024, 1), 0); del buf532  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_689, buf540, out=buf542)
    del permute_689
    buf547 = buf516; del buf516  # reuse
    cpp_fused__unsafe_view_clone_133(c_void_p(buf542.data_ptr()), c_void_p(buf547.data_ptr()))
    buf549 = reinterpret_tensor(buf542, (4096, 512), (512, 1), 0); del buf542  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf547, permute_700, out=buf549)
    del permute_700
    buf543 = reinterpret_tensor(buf513, (32, 1024, 64), (65536, 64, 1), 0); del buf513  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf540, permute_690, out=buf543)
    del permute_690
    buf550 = buf510; del buf510  # reuse
    cpp_fused_view_134(c_void_p(buf543.data_ptr()), c_void_p(buf550.data_ptr()))
    buf552 = reinterpret_tensor(buf543, (4096, 512), (512, 1), 0); del buf543  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf550, permute_705, out=buf552)
    del permute_705
    buf554 = buf528; del buf528  # reuse
    buf555 = buf487; del buf487  # reuse
    cpp_fused_add_div_mul_pow_sum_135(c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(embedding.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()))
    del primals_3
    buf557 = reinterpret_tensor(buf524, (4096, 2048), (2048, 1), 0); del buf524  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf555, (4096, 512), (512, 1), 0), permute_709, out=buf557)
    del permute_709
    buf558 = reinterpret_tensor(buf557, (4, 1024, 2048), (2097152, 2048, 1), 0); del buf557  # reuse
    cpp_fused_threshold_backward_136(c_void_p(buf558.data_ptr()), c_void_p(le_12.data_ptr()))
    del le_12
    buf560 = buf482; del buf482  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (4096, 2048), (2048, 1), 0), permute_713, out=buf560)
    del permute_713
    buf562 = buf554; del buf554  # reuse
    buf563 = reinterpret_tensor(buf479, (4, 1024, 512), (524288, 512, 1), 0); del buf479  # reuse
    cpp_fused_add_div_mul_pow_sum_137(c_void_p(buf560.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(embedding.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()))
    del primals_2
    buf565 = buf476; del buf476  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf563, (4096, 512), (512, 1), 0), permute_717, out=buf565)
    del permute_717
    buf566 = reinterpret_tensor(buf461, (4, 8, 1024, 64), (524288, 65536, 64, 1), 0); del buf461  # reuse
    cpp_fused_clone_138(c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()))
    buf567 = reinterpret_tensor(buf565, (32, 1024, 64), (65536, 64, 1), 0); del buf565  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_720, reinterpret_tensor(buf566, (32, 1024, 64), (65536, 64, 1), 0), out=buf567)
    del permute_720
    buf583 = buf492; del buf492  # reuse
    cpp_fused_view_139(c_void_p(buf567.data_ptr()), c_void_p(buf583.data_ptr()))
    buf585 = reinterpret_tensor(buf567, (4096, 512), (512, 1), 0); del buf567  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf583, permute_729, out=buf585)
    del permute_729
    buf568 = buf540; del buf540  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf566, (32, 1024, 64), (65536, 64, 1), 0), permute_721, out=buf568)
    del permute_721
    buf569 = buf535; del buf535  # reuse
    buf570 = reinterpret_tensor(buf568, (4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf568  # reuse
    buf571 = reinterpret_tensor(buf536, (33554432, ), (1, ), 0); del buf536  # reuse
    buf574 = reinterpret_tensor(buf355, (32, 1024, 1024), (1048576, 1024, 1), 0); del buf355  # reuse
    cpp_fused__softmax_backward_data_as_strided_scatter_140(c_void_p(buf570.data_ptr()), c_void_p(alias_122.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf574.data_ptr()))
    del alias_122
    del buf569
    del buf570
    buf581 = reinterpret_tensor(buf566, (32, 64, 1024), (65536, 1024, 1), 0); del buf566  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_723, buf574, out=buf581)
    del permute_723
    buf586 = buf484; del buf484  # reuse
    cpp_fused__unsafe_view_clone_141(c_void_p(buf581.data_ptr()), c_void_p(buf586.data_ptr()))
    buf588 = reinterpret_tensor(buf581, (4096, 512), (512, 1), 0); del buf581  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf586, permute_734, out=buf588)
    del permute_734
    buf582 = reinterpret_tensor(buf481, (32, 1024, 64), (65536, 64, 1), 0); del buf481  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf574, permute_724, out=buf582)
    del buf574
    del permute_724
    buf589 = buf478; del buf478  # reuse
    cpp_fused_view_142(c_void_p(buf582.data_ptr()), c_void_p(buf589.data_ptr()))
    buf591 = reinterpret_tensor(buf582, (4096, 512), (512, 1), 0); del buf582  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf589, permute_739, out=buf591)
    del permute_739
    buf527 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf553 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf561 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    buf592 = empty((1, 1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_143(c_void_p(buf526.data_ptr()), c_void_p(embedding.data_ptr()), c_void_p(mm_3.data_ptr()), c_void_p(mm_5.data_ptr()), c_void_p(mm_9.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf592.data_ptr()))
    del buf526
    del buf546
    del buf549
    del buf552
    del buf560
    del mm_3
    del mm_5
    del mm_9
    del rsqrt_1
    del rsqrt_2
    del rsqrt_3
    buf530 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (512, 4096), (1, 512), 0), view_43, out=buf530)
    del buf529
    del view_43
    buf545 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf544, (512, 4096), (1, 512), 0), view_25, out=buf545)
    del buf544
    buf548 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf547, (512, 4096), (1, 512), 0), view_25, out=buf548)
    del buf547
    buf551 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (512, 4096), (1, 512), 0), view_25, out=buf551)
    del buf550
    del view_25
    buf556 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf555, (512, 4096), (1, 512), 0), view_23, out=buf556)
    del buf555
    del view_23
    buf559 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (2048, 4096), (1, 2048), 0), view_21, out=buf559)
    del view_21
    buf564 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf563, (512, 4096), (1, 512), 0), view_19, out=buf564)
    del view_19
    buf576 = reinterpret_tensor(buf558, (1, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0); del buf558  # reuse
    buf578 = reinterpret_tensor(buf576, (1024, 1024, 8), (1024, 1, 1048576), 0); del buf576  # reuse
    buf577 = empty((32, 8), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_sum_threshold_backward_144(c_void_p(buf578.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf577.data_ptr()))
    del buf401
    del buf435
    del buf469
    del buf503
    del buf537
    del buf571
    aten.index_put_(buf577, [add_3], buf578, True)
    del add_3
    del buf578
    buf584 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf583, (512, 4096), (1, 512), 0), view_1, out=buf584)
    del buf583
    buf587 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf586, (512, 4096), (1, 512), 0), view_1, out=buf587)
    del buf586
    buf590 = empty((512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf589, (512, 4096), (1, 512), 0), view_1, out=buf590)
    del buf589
    del view_1
    buf593 = buf562; del buf562  # reuse
    buf594 = buf563; del buf563  # reuse
    buf595 = empty((32128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_mul_pow_sum_threshold_backward_145(c_void_p(buf594.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(embedding.data_ptr()), c_void_p(view.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()))
    del buf585
    del buf588
    del buf591
    del buf593
    del embedding
    del primals_1
    del rsqrt
    aten.index_put_(buf595, [view], buf594, True)
    del buf594
    del view
    buf382 = empty((32128, 512), device='cpu', dtype=torch.float32)
    buf598 = buf382; del buf382  # reuse
    cpp_fused_add_146(c_void_p(buf598.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf595.data_ptr()))
    return (reinterpret_tensor(buf592, (512, ), (1, ), 0), reinterpret_tensor(buf561, (512, ), (1, ), 0), reinterpret_tensor(buf553, (512, ), (1, ), 0), reinterpret_tensor(buf527, (512, ), (1, ), 0), reinterpret_tensor(buf519, (512, ), (1, ), 0), reinterpret_tensor(buf493, (512, ), (1, ), 0), reinterpret_tensor(buf485, (512, ), (1, ), 0), reinterpret_tensor(buf459, (512, ), (1, ), 0), reinterpret_tensor(buf451, (512, ), (1, ), 0), reinterpret_tensor(buf425, (512, ), (1, ), 0), reinterpret_tensor(buf417, (512, ), (1, ), 0), reinterpret_tensor(buf391, (512, ), (1, ), 0), reinterpret_tensor(buf383, (512, ), (1, ), 0), reinterpret_tensor(buf377, (512, ), (1, ), 0), reinterpret_tensor(buf346, (512, ), (1, ), 0), reinterpret_tensor(buf319, (512, ), (1, ), 0), reinterpret_tensor(buf311, (512, ), (1, ), 0), reinterpret_tensor(buf285, (512, ), (1, ), 0), reinterpret_tensor(buf259, (512, ), (1, ), 0), reinterpret_tensor(buf251, (512, ), (1, ), 0), reinterpret_tensor(buf225, (512, ), (1, ), 0), reinterpret_tensor(buf198, (512, ), (1, ), 0), reinterpret_tensor(buf190, (512, ), (1, ), 0), reinterpret_tensor(buf164, (512, ), (1, ), 0), reinterpret_tensor(buf138, (512, ), (1, ), 0), reinterpret_tensor(buf130, (512, ), (1, ), 0), reinterpret_tensor(buf104, (512, ), (1, ), 0), reinterpret_tensor(buf78, (512, ), (1, ), 0), reinterpret_tensor(buf70, (512, ), (1, ), 0), reinterpret_tensor(buf44, (512, ), (1, ), 0), reinterpret_tensor(buf17, (512, ), (1, ), 0), reinterpret_tensor(buf9, (512, ), (1, ), 0), buf598, reinterpret_tensor(buf590, (512, 512), (512, 1), 0), reinterpret_tensor(buf587, (512, 512), (512, 1), 0), reinterpret_tensor(buf584, (512, 512), (512, 1), 0), buf577, reinterpret_tensor(buf564, (512, 512), (512, 1), 0), reinterpret_tensor(buf559, (2048, 512), (512, 1), 0), reinterpret_tensor(buf556, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf551, (512, 512), (512, 1), 0), reinterpret_tensor(buf548, (512, 512), (512, 1), 0), reinterpret_tensor(buf545, (512, 512), (512, 1), 0), reinterpret_tensor(buf530, (512, 512), (512, 1), 0), reinterpret_tensor(buf525, (2048, 512), (512, 1), 0), reinterpret_tensor(buf522, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf517, (512, 512), (512, 1), 0), reinterpret_tensor(buf514, (512, 512), (512, 1), 0), reinterpret_tensor(buf511, (512, 512), (512, 1), 0), reinterpret_tensor(buf496, (512, 512), (512, 1), 0), reinterpret_tensor(buf491, (2048, 512), (512, 1), 0), reinterpret_tensor(buf488, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf483, (512, 512), (512, 1), 0), reinterpret_tensor(buf480, (512, 512), (512, 1), 0), reinterpret_tensor(buf477, (512, 512), (512, 1), 0), reinterpret_tensor(buf462, (512, 512), (512, 1), 0), reinterpret_tensor(buf457, (2048, 512), (512, 1), 0), reinterpret_tensor(buf454, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf449, (512, 512), (512, 1), 0), reinterpret_tensor(buf446, (512, 512), (512, 1), 0), reinterpret_tensor(buf443, (512, 512), (512, 1), 0), reinterpret_tensor(buf428, (512, 512), (512, 1), 0), reinterpret_tensor(buf423, (2048, 512), (512, 1), 0), reinterpret_tensor(buf420, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf415, (512, 512), (512, 1), 0), reinterpret_tensor(buf412, (512, 512), (512, 1), 0), reinterpret_tensor(buf409, (512, 512), (512, 1), 0), reinterpret_tensor(buf394, (512, 512), (512, 1), 0), reinterpret_tensor(buf389, (2048, 512), (512, 1), 0), reinterpret_tensor(buf386, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf375, (512, 512), (512, 1), 0), reinterpret_tensor(buf372, (512, 512), (512, 1), 0), reinterpret_tensor(buf369, (512, 512), (512, 1), 0), buf362, reinterpret_tensor(buf349, (512, 512), (512, 1), 0), reinterpret_tensor(buf344, (512, 512), (512, 1), 0), reinterpret_tensor(buf340, (512, 512), (512, 1), 0), reinterpret_tensor(buf337, (512, 512), (512, 1), 0), reinterpret_tensor(buf322, (512, 512), (512, 1), 0), reinterpret_tensor(buf317, (2048, 512), (512, 1), 0), reinterpret_tensor(buf314, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf309, (512, 512), (512, 1), 0), reinterpret_tensor(buf306, (512, 512), (512, 1), 0), reinterpret_tensor(buf303, (512, 512), (512, 1), 0), reinterpret_tensor(buf288, (512, 512), (512, 1), 0), reinterpret_tensor(buf283, (512, 512), (512, 1), 0), reinterpret_tensor(buf280, (512, 512), (512, 1), 0), reinterpret_tensor(buf277, (512, 512), (512, 1), 0), reinterpret_tensor(buf262, (512, 512), (512, 1), 0), reinterpret_tensor(buf257, (2048, 512), (512, 1), 0), reinterpret_tensor(buf254, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf249, (512, 512), (512, 1), 0), reinterpret_tensor(buf246, (512, 512), (512, 1), 0), reinterpret_tensor(buf243, (512, 512), (512, 1), 0), reinterpret_tensor(buf228, (512, 512), (512, 1), 0), reinterpret_tensor(buf223, (512, 512), (512, 1), 0), reinterpret_tensor(buf219, (512, 512), (512, 1), 0), reinterpret_tensor(buf216, (512, 512), (512, 1), 0), reinterpret_tensor(buf201, (512, 512), (512, 1), 0), reinterpret_tensor(buf196, (2048, 512), (512, 1), 0), reinterpret_tensor(buf193, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf188, (512, 512), (512, 1), 0), reinterpret_tensor(buf185, (512, 512), (512, 1), 0), reinterpret_tensor(buf182, (512, 512), (512, 1), 0), reinterpret_tensor(buf167, (512, 512), (512, 1), 0), reinterpret_tensor(buf162, (512, 512), (512, 1), 0), reinterpret_tensor(buf159, (512, 512), (512, 1), 0), reinterpret_tensor(buf156, (512, 512), (512, 1), 0), reinterpret_tensor(buf141, (512, 512), (512, 1), 0), reinterpret_tensor(buf136, (2048, 512), (512, 1), 0), reinterpret_tensor(buf133, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf128, (512, 512), (512, 1), 0), reinterpret_tensor(buf125, (512, 512), (512, 1), 0), reinterpret_tensor(buf122, (512, 512), (512, 1), 0), reinterpret_tensor(buf107, (512, 512), (512, 1), 0), reinterpret_tensor(buf102, (512, 512), (512, 1), 0), reinterpret_tensor(buf99, (512, 512), (512, 1), 0), reinterpret_tensor(buf96, (512, 512), (512, 1), 0), reinterpret_tensor(buf81, (512, 512), (512, 1), 0), reinterpret_tensor(buf76, (2048, 512), (512, 1), 0), reinterpret_tensor(buf73, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf68, (512, 512), (512, 1), 0), reinterpret_tensor(buf65, (512, 512), (512, 1), 0), reinterpret_tensor(buf62, (512, 512), (512, 1), 0), reinterpret_tensor(buf47, (512, 512), (512, 1), 0), reinterpret_tensor(buf42, (512, 512), (512, 1), 0), reinterpret_tensor(buf39, (512, 512), (512, 1), 0), reinterpret_tensor(buf36, (512, 512), (512, 1), 0), reinterpret_tensor(buf20, (512, 512), (512, 1), 0), reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(buf12, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf7, (32128, 512), (512, 1), 0), None, None, )


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
    view = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    embedding = rand_strided((4, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_3 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    view_19 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_3 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_1 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_5 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_2 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_43 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_9 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_3 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_11 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_4 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_15 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_5 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_17 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_21 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_7 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_93 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_23 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_8 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_27 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_9 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_29 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_10 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_121 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_139 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_33 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_11 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_35 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_12 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((4, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    embedding_2 = rand_strided((4, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    rsqrt_13 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_146 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    view_164 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_39 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_14 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_166 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_169 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_184 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_43 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_15 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_186 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_188 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_45 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_190 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_208 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_49 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_17 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_210 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_228 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_53 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_230 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_232 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_55 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_19 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_234 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_252 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_59 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_20 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_254 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_272 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_63 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_21 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_274 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_276 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_65 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_22 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_278 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_296 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_69 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_23 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_298 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_316 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_73 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_24 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_318 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_320 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_75 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_322 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_340 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_79 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_26 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_342 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_360 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_83 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_27 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_362 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_364 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_85 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_28 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_366 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_384 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_89 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_29 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_386 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_404 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    mm_93 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_30 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_406 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_408 = rand_strided((4096, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mm_95 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    rsqrt_31 = rand_strided((4, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    view_410 = rand_strided((4096, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_1 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_199 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_207 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_65 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_231 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_232 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_67 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_2 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_257 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_71 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_272 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_73 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_297 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_3 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_315 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_77 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_330 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_347 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_79 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_4 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_373 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_83 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_402 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_406 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_85 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_407 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_423 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_427 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_5 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_431 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_89 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_441 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_451 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_456 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_463 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_91 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_465 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_466 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_485 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_6 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_489 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_493 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_496 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_497 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_95 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_498 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_499 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_504 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_518 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_521 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_522 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_97 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_524 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_535 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_540 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_544 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_7 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_548 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_552 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_555 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_556 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_102 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_557 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_558 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_563 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_568 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_573 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_577 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_8 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_581 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_585 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_588 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_589 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_106 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_590 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_591 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_596 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_601 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_606 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_610 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_9 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_614 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_618 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_621 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_622 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_110 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_623 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_624 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_629 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_634 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_639 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_643 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_10 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_647 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_651 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_654 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_655 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_114 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_656 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_657 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_662 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_667 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_672 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_676 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_11 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_680 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_684 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_687 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_688 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_118 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_689 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_690 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_695 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_700 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_705 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_709 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    le_12 = rand_strided((4, 1024, 2048), (2097152, 2048, 1), device='cpu', dtype=torch.bool)
    permute_713 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_717 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_720 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_721 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    alias_122 = rand_strided((4, 8, 1024, 1024), (8388608, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_723 = rand_strided((32, 64, 1024), (65536, 1, 64), device='cpu', dtype=torch.float32)
    permute_724 = rand_strided((32, 1024, 64), (65536, 1, 1024), device='cpu', dtype=torch.float32)
    permute_729 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_734 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_739 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1024, 32128), (32899072, 32128, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((4, 8, 1024, 64), (524288, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((4, 1024, 512), (524288, 512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, view, embedding, rsqrt, view_1, add_3, view_19, mm_3, rsqrt_1, view_21, view_23, mm_5, rsqrt_2, view_25, view_43, mm_9, rsqrt_3, view_45, view_47, mm_11, rsqrt_4, view_49, view_67, mm_15, rsqrt_5, view_69, view_71, mm_17, rsqrt_6, view_73, view_91, mm_21, rsqrt_7, view_93, view_95, mm_23, rsqrt_8, view_97, view_115, mm_27, rsqrt_9, view_117, view_119, mm_29, rsqrt_10, view_121, view_139, mm_33, rsqrt_11, view_141, view_143, mm_35, rsqrt_12, view_145, embedding_2, rsqrt_13, view_146, add_37, view_164, mm_39, rsqrt_14, view_166, view_169, view_184, mm_43, rsqrt_15, view_186, view_188, mm_45, rsqrt_16, view_190, view_208, mm_49, rsqrt_17, view_210, view_228, mm_53, rsqrt_18, view_230, view_232, mm_55, rsqrt_19, view_234, view_252, mm_59, rsqrt_20, view_254, view_272, mm_63, rsqrt_21, view_274, view_276, mm_65, rsqrt_22, view_278, view_296, mm_69, rsqrt_23, view_298, view_316, mm_73, rsqrt_24, view_318, view_320, mm_75, rsqrt_25, view_322, view_340, mm_79, rsqrt_26, view_342, view_360, mm_83, rsqrt_27, view_362, view_364, mm_85, rsqrt_28, view_366, view_384, mm_89, rsqrt_29, view_386, view_404, mm_93, rsqrt_30, view_406, view_408, mm_95, rsqrt_31, view_410, permute_191, permute_195, le_1, permute_199, permute_203, permute_206, permute_207, alias_65, permute_208, permute_209, permute_214, permute_219, permute_224, permute_228, permute_231, permute_232, alias_67, permute_233, permute_234, permute_239, permute_244, permute_249, permute_253, le_2, permute_257, permute_261, permute_264, permute_265, alias_71, permute_266, permute_267, permute_272, permute_277, permute_282, permute_286, permute_289, permute_290, alias_73, permute_291, permute_292, permute_297, permute_302, permute_307, permute_311, le_3, permute_315, permute_319, permute_322, permute_323, alias_77, permute_324, permute_325, permute_330, permute_335, permute_340, permute_344, permute_347, permute_348, alias_79, permute_349, permute_350, permute_355, permute_360, permute_365, permute_369, le_4, permute_373, permute_377, permute_380, permute_381, alias_83, permute_382, permute_383, permute_388, permute_393, permute_398, permute_402, permute_405, permute_406, alias_85, permute_407, permute_408, permute_413, permute_418, permute_423, permute_427, le_5, permute_431, permute_435, permute_438, permute_439, alias_89, permute_440, permute_441, permute_446, permute_451, permute_456, permute_460, permute_463, permute_464, alias_91, permute_465, permute_466, permute_471, permute_476, permute_481, permute_485, le_6, permute_489, permute_493, permute_496, permute_497, alias_95, permute_498, permute_499, permute_504, permute_509, permute_514, permute_518, permute_521, permute_522, alias_97, permute_524, permute_525, permute_530, permute_535, permute_540, permute_544, le_7, permute_548, permute_552, permute_555, permute_556, alias_102, permute_557, permute_558, permute_563, permute_568, permute_573, permute_577, le_8, permute_581, permute_585, permute_588, permute_589, alias_106, permute_590, permute_591, permute_596, permute_601, permute_606, permute_610, le_9, permute_614, permute_618, permute_621, permute_622, alias_110, permute_623, permute_624, permute_629, permute_634, permute_639, permute_643, le_10, permute_647, permute_651, permute_654, permute_655, alias_114, permute_656, permute_657, permute_662, permute_667, permute_672, permute_676, le_11, permute_680, permute_684, permute_687, permute_688, alias_118, permute_689, permute_690, permute_695, permute_700, permute_705, permute_709, le_12, permute_713, permute_717, permute_720, permute_721, alias_122, permute_723, permute_724, permute_729, permute_734, permute_739, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_T5', benchmark_compiled_module)
